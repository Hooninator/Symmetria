#ifndef LOCAL_MULT_CUH
#define LOCAL_MULT_CUH

#include "../common.h"
#include "dCSR.cuh"
#include "SemiRingInterface.h"
#include "../source/device/Multiply.cuh"

#include "transpose_csr.cuh"
#include "dCSR_utils.cuh"


namespace symmetria {

    /*
template <typename IT, typename DT>
struct DeviceTriple 
{

    IT row, col;
    DT val;

    template <typename IT2, typename DT2>
    friend std::ostream& operator<< (std::ostream& os, DeviceTriple<IT2, DT2>& t);

};


template <typename IT, typename DT>
std::ostream& operator<<(std::ostream& os, DeviceTriple<IT, DT>& t)
{
    os<<"("<<t.row<<","<<t.col<<","<<t.val<<")"<<std::endl;
    return os;
}

*/


template <typename IT, typename IT2, typename DT>
__global__ void dCSR_to_triples(DT * d_vals, IT * d_colinds, IT * d_rowptrs, 
                                std::tuple<IT, IT, DT> * d_triples,
                                const IT2 rows)
{
    const uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    const uint32_t wid = tid / warpSize;
    const uint32_t lid = tid % warpSize;

    if (wid < rows) {
        IT start = d_rowptrs[wid];
        IT end = d_rowptrs[wid+1];
        for (int j = start + lid; j < end; j += warpSize)
        {
            std::get<0>(d_triples[j]) = wid;
            std::get<1>(d_triples[j]) = d_colinds[j];
            std::get<2>(d_triples[j]) = d_vals[j];
        }
    }
}


/* Call GALATIC SpGEMM, convert output to tuples, return pointer to tuples on device */
template <typename SR, typename IT, typename DT>
std::tuple<IT, IT, DT> * local_spgemm_galatic(dCSR<DT>& A, dCSR<DT>& A_t, IT& nnz)
{

    using Triple = std::tuple<IT, IT, DT>;


    //Note: I don't know what any of this is
    const int Threads = 128;
    const int BlocksPerMP = 1;
    const int NNZPerThread = 2;
    const int InputElementsPerThreads = 2;
    const int RetainElementsPerThreads = 1;
    const int MaxChunksToMerge = 16;
    const int MaxChunksGeneralizedMerge = 256; // MAX: 865
    const int MergePathOptions = 8;
    
    
    GPUMatrixMatrixMultiplyTraits DefaultTraits(Threads, BlocksPerMP, NNZPerThread,
                                                 InputElementsPerThreads, RetainElementsPerThreads,
                                                 MaxChunksToMerge, MaxChunksGeneralizedMerge, 
                                                 MergePathOptions );
    ExecutionStats stats; //TODO: Do I have to have this
    dCSR<DT> C;
    SR semiring;

    /* Do multiply */
    ACSpGEMM::Multiply<SR>(A, A_t, C, DefaultTraits, stats, true, semiring);
    CUDA_CHECK(cudaDeviceSynchronize());
    nnz = C.nnz;

    /* Convert to device triples */
    Triple * d_triples;
    CUDA_CHECK(cudaMalloc(&d_triples, sizeof(Triple)*nnz));
    const uint32_t tpb = 256;
    const uint32_t wpb = tpb / 32;
    const uint32_t blocks = std::ceil( static_cast<double>(C.rows) / static_cast<double>(wpb) );
    dCSR_to_triples<<<blocks, tpb>>>(C.data, C.col_ids, C.row_offsets, d_triples, C.rows);
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_triples;
}



}

#endif
