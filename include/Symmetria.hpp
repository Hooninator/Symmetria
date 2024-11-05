#ifndef SYMMETRIA_HPP
#define SYMMETRIA_HPP

#include "common.h"
#include "DistSpMat.hpp"
#include "DistSpMatCyclic.hpp"
#include "matrix_market_io.h"
#include "semirings.cuh"

#include "kernels/spsyrk.cuh"


namespace symmetria {


void symmetria_init()
{
    /* MPI */
    MPI_Init(nullptr, nullptr);

    /* NVSHMEM */
    MPI_Comm comm = MPI_COMM_WORLD;
    attr.mpi_comm = &(comm);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    std::cout<<"Doing it"<<std::endl;

    /* OpenSHMEM */
    //shmem_init();

    my_pe = nvshmem_my_pe();
    n_pes = nvshmem_n_pes();

    my_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

	cudaSetDevice(my_pe_node);
    std::cout<<"Doing it again"<<std::endl;

    /* cuSPARSE */
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));

    /* Logfiles */
#ifdef DEBUG
    logptr = new Log(my_pe);
#endif

    /* Timers */
    timer_ptr = new Timer();

}


void symmetria_finalize()
{
    shmem_finalize();
    nvshmem_finalize();

    MPI_Finalize();

    CUSPARSE_CHECK(cusparseDestroy(cusparse_handle));

#ifdef DEBUG
    delete logptr;
#endif

    delete timer_ptr;

}

}


#endif
