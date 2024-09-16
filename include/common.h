#ifndef COMMON_H
#define COMMON_H

#include <unordered_map>
#include <string>
#include <fstream>
#include <iostream>

#include <mpi.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#include <shmem.h>
#include <shmemx.h>

#include <cusparse_v2.h>

#include "utils/Timer.hpp"
#include "utils/Log.hpp"
#include "utils/colors.h"

#define CUDA_CHECK(call) {                                                 \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " ("     \
                  << __FILE__ << ":" << __LINE__ << ")" << std::endl;      \
        exit(err);                                                         \
    }                                                                      \
}

#define CUSPARSE_CHECK(call) do {                                    \
    cusparseStatus_t err = call;                                     \
    if (err != CUSPARSE_STATUS_SUCCESS) {                            \
        fprintf(stderr, "cuSPARSE error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cusparseGetErrorString(err));    \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while(0)

#define ERROR(msg) {std::cerr<<msg<<std::endl; exit(1);}


namespace symmetria {

    int my_pe;
    int my_pe_node;
    int n_pes;
    int n_pes_node;

    Timer * timer;
    Log * logptr;

    cusparseHandle_t cusparse_handle;

    nvshmemx_init_attr_t attr;

}


#endif
