#ifndef SPSYRK_CUH
#define SPSYRK_CUH

#include "../common.h"
#include "../DistSpMat.hpp"

#include "local_mult.cuh"
#include "dCSR_utils.cuh"
#include "transpose_csr.cuh"
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
        MPI_Comm_create(world, world_group, &comms[i]);
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
    CooTriples<IT, DT> C_products; 
        
    DEBUG_PRINT("Beginning main loop");

    /* Main loop */
    for (int k=0; k<p; k++)
    {

        if (k > rank) break;

        MPI_Request * requests = new MPI_Request[3];

        /* Allocate space for broadcast tiles */
        if (k==rank) {
            A_recv = A_t_loc;
        } else {
            A_recv.alloc(A.get_loc_cols(), A.get_loc_rows(), A.get_tile_sizes()[k]);
        }

        /* Non-blocking broadcast of tranposed block row k */
        mpi::ibcast_tile(k, comms[k], 
                          A_recv.data, A_recv.col_ids, A_recv.row_offsets,
                          requests, A_recv.nnz, A_recv.rows);

        /* If first iteration, overlap broadcasts with computing diagonal blocks */
        //TODO

        /* Wait on bcast completion */
        MPI_Waitall(3, requests, MPI_STATUSES_IGNORE);

#ifdef DEBUG
        logptr->OFS()<<"Received round "<<k<<std::endl;
        dump_dCSR_to_log(logptr, A_recv);
#endif

        /* If rank > k, multiply the tile I just received */
        IT nnzC;
        auto d_C_acc = local_spgemm_galatic<SR>(A_loc, A_recv, nnzC);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Push output tuples to CooTriples on host */
        std::tuple<IT, IT, DT> * C_to_add = new std::tuple<IT, IT, DT>[nnzC];
        CUDA_CHECK(cudaMemcpy(C_to_add, d_C_acc, sizeof(std::tuple<IT, IT, DT>)*nnzC,
                                cudaMemcpyDeviceToHost));
        C_products.add_triples(C_to_add, nnzC);

        /* Cleanup */
        delete requests;
        A_recv.reset();
        CUDA_CHECK(cudaFree(d_C_acc));

    }

    MPI_Barrier(world);

    DEBUG_PRINT("Out of main loop");

#ifdef DEBUG
    C_products.dump_to_log(logptr, "Output tuples prior to merging");
#endif

    /* Merge output tuples */
    //TODO

    /* Cleanup */
    MPI_Group_free(&world_group);

    /* Return final matrix */
    DistSpMat1DBlockRow<IT, DT> C(A.get_rows(), A.get_rows(), C_products.get_nnz(),
                                    proc_map);
    C.set_from_coo(&C_products);

    return C;
}

}


#endif
