#ifndef SPSYRK_CUH
#define SPSYRK_CUH

#include "../common.h"
#include "../DistSpMat.hpp"
#include "../DistSpMatCyclic.hpp"

#include "local_mult.cuh"
#include "transpose_csr.cuh"
#include "merge.cuh"
#include "mpi_utils.cuh"


namespace symmetria {
template <typename SR, typename IT, typename DT>
DistSpMat1DBlockRow<IT, DT> spsyrk_bulksync_1d_rowblock(DistSpMat1DBlockRow<IT, DT>& A)
{
        
    DEBUG_PRINT("Starting spsyrk 1d");

    /* Bookkeeping */
    auto proc_map = A.proc_map;
    const int p = proc_map->get_n_procs();
    int rank = proc_map->get_rank();
    MPI_Comm world = proc_map->get_world_comm();
    
    /* Need array of communicators for each subset of the processes that are broadcast to at each stage */
    std::vector<MPI_Comm> comms(p);

    std::vector<int> procs_in_group(p);
    std::iota(procs_in_group.begin(), procs_in_group.end(), 0);

    MPI_Group world_group;
    MPI_Comm_group(world, &world_group);

    for (int i=0; i<p; i++)
    {
        MPI_Group group;
        MPI_Group_incl(world_group, (p - i), procs_in_group.data() + i, &group);
        MPI_Comm_create(world, group, &comms[i]);
    }

    MPI_Barrier(world);

    DEBUG_PRINT("Done with bookkeeping setup");

    /* My local block */
    auto A_loc = make_dCSR_from_distspmat<DT>(A);

    /* Create transposed version of my local block row */
    auto A_t_loc = transpose_outofplace(A_loc);

    /* This matrix will store tiles brodcast by other processes */
    dCSR<DT> A_recv;

    /* COO triples on host used to store C */
    std::vector<std::tuple<IT, IT, DT>*> C_products;
    IT * C_nnz_arr = new IT[rank+1];
        
    DEBUG_PRINT("Beginning main loop");

    /* Main loop */
    for (int k=0; k<p; k++)
    {
        if (k > rank) break;

#ifdef DEBUG
        logptr->OFS()<<"Beginning iteration "<<k<<std::endl;
#endif

        MPI_Request * requests = new MPI_Request[3];

        /* Allocate space for broadcast tiles */
        if (k==rank) {
            A_recv = A_t_loc;
#ifdef DEBUG_LOG
            logptr->OFS()<<"Sending round "<<k<<std::endl;
            dump_dCSR_to_log(logptr, A_recv);
#endif
        } else {
            A_recv.alloc(A.get_loc_cols(), A.get_loc_rows(), A.get_tile_sizes()[k]);
        }

#ifdef TIMING
        timer_ptr->start_timer("Broadcast");
#endif

        /* Non-blocking broadcast of tranposed block row k */
        //TODO: Pack every buffer into one sendbuf and unpack it on the receiver
        mpi::ibcast_tile(0, comms[k], //always root 0 because we're not using comm world
                          A_recv.data, A_recv.col_ids, A_recv.row_offsets,
                          requests, A_recv.nnz, A_recv.rows);

        /* If first iteration, overlap broadcasts with computing diagonal blocks */
        //TODO

        /* Wait on bcast completion */
        MPI_Waitall(3, requests, MPI_STATUSES_IGNORE);

#ifdef TIMING
        timer_ptr->stop_timer("Broadcast");
#endif

#ifdef DEBUG_
        logptr->OFS()<<"Received round "<<k<<std::endl;
        dump_dCSR_to_log(logptr, A_recv);
#endif

#ifdef TIMING
        timer_ptr->start_timer("LocalMultiply");
#endif

        /* If rank > k, multiply the tile I just received */
        IT nnzC;
        IT offset = k * (A.get_rows() / p);
        auto d_C_acc = local_spgemm_galatic<SR>(A_loc, A_recv, nnzC, offset);
        CUDA_CHECK(cudaDeviceSynchronize());
        C_nnz_arr[k] = nnzC;

#ifdef TIMING
        timer_ptr->stop_timer("LocalMultiply");
#endif

#ifdef TIMING
        timer_ptr->start_timer("CopyTriples");
#endif

        /* Push output tuples to CooTriples on host */
        std::tuple<IT, IT, DT> * C_to_add = new std::tuple<IT, IT, DT>[nnzC];
        CUDA_CHECK(cudaMemcpy(C_to_add, d_C_acc, sizeof(std::tuple<IT, IT, DT>)*nnzC, cudaMemcpyDeviceToHost));
        C_products.push_back(C_to_add);

#ifdef TIMING
        timer_ptr->stop_timer("CopyTriples");
#endif

        /* Cleanup */
        delete requests;
        if (k != rank) {
            A_recv.reset();
        }
        CUDA_CHECK(cudaFree(d_C_acc));
    }

    MPI_Barrier(world);

    DEBUG_PRINT("Out of main loop");

#ifdef TIMING
    timer_ptr->start_timer("Merge");
#endif

    /* Merge output tuples */
	SR semiring;
    auto C_final = merge_hash_combblas<SR>(C_products, C_nnz_arr, 
                                           A.get_loc_rows(), A.get_rows());

    DEBUG_PRINT("Done with merge");

#ifdef TIMING
    timer_ptr->stop_timer("Merge");
#endif

#ifdef DEBUG
    C_final.dump_to_log(logptr, "Final output");
#endif

    /* Compute final nnz */
    IT total_nnz = C_final.get_nnz();
    MPI_Allreduce(MPI_IN_PLACE, &total_nnz, 1, MPIType<IT>(), MPI_SUM, world);

    /* Cleanup */
    MPI_Group_free(&world_group);
    clear_dCSR_ptrs(A_loc); //necessary to prevent destructor from freeing A
    clear_dCSR_ptrs(A_recv);
    delete[] C_nnz_arr;

#ifdef TIMING
    timer_ptr->start_timer("OutputConstruction");
#endif

    /* Return final matrix */
    DistSpMat1DBlockRow<IT, DT> C(A.get_rows(), A.get_rows(), total_nnz,
                                    proc_map);
    C.set_from_coo(&C_final);

#ifdef TIMING
    timer_ptr->stop_timer("OutputConstruction");
#endif

    return C;
}




template <typename SR, typename IT, typename DT, typename P>
DistSpMatCyclic2D<IT, DT, P> spsyrk_cyclic_2d(DistSpMatCyclic2D<IT, DT, P>& A)
{

    /* Bookkeeping */
    auto tile_window = A.get_tile_window();
    auto proc_map = A.proc_map;
    int np = proc_map->get_n_procs();
    int rank = proc_map->get_rank();
    auto my_tile_inds = proc_map->get_my_tile_inds();

    std::vector<CooTriples<IT, DT>> C_tiles(my_tile_inds.size());
    IT total_nnz = 0;
    
    /* Main loop */
    for (int tile_id = 0; tile_id<my_tile_inds.size(); tile_id++)
    {
        auto tile_inds = my_tile_inds[tile_id];
        int i = tile_inds.first;
        int j = tile_inds.second;

        /* To store output tuples prior to merging */
        std::vector<std::tuple<IT, IT, DT>*> C_products;
        std::vector<IT> C_nnz_arr;
        

        if (i < j) continue; //Skip if in strictly upper triangular region

        for (int k=0; k<proc_map->get_ntiles(); k++)
        {
            /* Get kth tile in prow i and prow j */

            SpMat<IT, DT> A_tile = A.get_tile_sync(i, k);
            SpMat<IT, DT> B_tile = A.get_tile_sync(j, k);

            if (A_tile.get_nnz()==0 || B_tile.get_nnz()==0)
            {
                A_tile.free();
                B_tile.free();
                continue;
            }

#ifdef DEBUG
            //A_tile.dump_to_log(logptr, "A_tile");
            //B_tile.dump_to_log(logptr, "B_tile");
#endif

            IT my_nnz = 0;
            auto C_tuples = local_spgemm_galatic<SR>(A_tile, B_tile, my_nnz);
            CUDA_CHECK(cudaDeviceSynchronize());

            C_nnz_arr.push_back(my_nnz);

            std::tuple<IT, IT, DT> * C_to_add = new std::tuple<IT, IT, DT>[my_nnz];
            CUDA_CHECK(cudaMemcpy(C_to_add, C_tuples, sizeof(std::tuple<IT, IT, DT>)*my_nnz, 
                                    cudaMemcpyDeviceToHost));
            C_products.push_back(C_to_add);

            A_tile.free();
            B_tile.free();
            CUDA_FREE_SAFE(C_tuples);

        }

        auto C_tile = merge_hash_combblas<SR>(C_products, C_nnz_arr.data(), A.get_mb(), A.get_mb());
        total_nnz += C_tile.get_nnz();


        ///* TODO: add a copy constructor to CooTriples and change this to emplace_back */
        C_tiles[tile_id] = C_tile;

    }

    DEBUG_PRINT("Out of main loop");

    MPI_Allreduce(MPI_IN_PLACE, &total_nnz, 1, MPIType<IT>(), MPI_SUM, proc_map->get_world_comm());
    DEBUG_PRINT("Total nnz: " + STR(total_nnz));

    nvshmem_barrier_all();

#ifdef DEBUG
    logptr->OFS()<<"C tiles"<<std::endl;
    std::for_each(C_tiles.begin(), C_tiles.end(), 
            [&](auto& C_tile) {C_tile.dump_to_log(logptr);});
#endif

    DistSpMatCyclic2D<IT, DT, P> C(A.get_rows(), A.get_rows(), total_nnz, A.get_mb(), A.get_mb(), proc_map);
    C.set_from_coo(C_tiles);

    MPI_Barrier(proc_map->get_world_comm());
    DEBUG_PRINT("DONE");

    return C;

}


}


#endif
