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

    /* OpenSHMEM */
    shmem_init();

    MPI_Comm_rank(MPI_COMM_WORLD, &my_pe);
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

    cudaGetDeviceCount(&n_pes_node);

    my_pe_node = my_pe % n_pes_node;

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
    MPI_Finalize();

    CUSPARSE_CHECK(cusparseDestroy(cusparse_handle));

#ifdef DEBUG
    delete logptr;
#endif

    delete timer_ptr;

}

}


#endif
