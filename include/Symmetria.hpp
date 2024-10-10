#ifndef SYMMETRIA_HPP
#define SYMMETRIA_HPP

#include "common.h"
#include "DistSpMat.hpp"
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

    /* NVSHMEM */
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    my_pe = nvshmem_my_pe();
    n_pes = nvshmem_n_pes();

    cudaGetDeviceCount(&n_pes_node);

    my_pe_node = my_pe % n_pes_node;

    /* cuSPARSE */
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));

    //TODO: GALATIC?

    /* Logfiles */
#ifdef DEBUG
    logptr = new Log(my_pe);
#endif

    /* Timers */
    timer_ptr = new Timer();

}


void symmetria_finalize()
{
    nvshmem_finalize();
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
