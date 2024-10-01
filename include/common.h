#ifndef COMMON_H
#define COMMON_H

#include <unordered_map>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <memory>
#include <algorithm>
#include <numeric>

#include <mpi.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#include <shmem.h>
#include <shmemx.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include "utils/Timer.hpp"
#include "utils/Log.hpp"
#include "utils/colors.h"
#include "utils/MPITypes.h"

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

#define NVSHMEM_FREE_SAFE(ptr) {if (ptr!=nullptr) nvshmem_free(ptr);}


namespace symmetria {


/* GLOBALS */
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