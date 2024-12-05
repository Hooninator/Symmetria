#ifndef LOCAL_MULT_CUH
#define LOCAL_MULT_CUH

#include "../common.h"
#include "../dCSR_utils.cuh"
#include "dCSR.cuh"
#include "CSR.cuh"
#include "CPU_SpGEMM.h"
#include "SemiRingInterface.h"
#include "../source/device/Multiply.cuh"

#include "transpose_csr.cuh"

#include <thrust/count.h>
#include <thrust/copy.h>

namespace symmetria {

template <typename IT, typename DT>
struct DeviceTuples
{
    IT row, col;
    DT val;
};


template <typename IT, typename DT>
std::ofstream& operator<<(std::ofstream& os, DeviceTuples<IT, DT>& t)
{
    os<<"("<<t.row<<","<<t.col<<","<<t.val<<")";
    return os;
}


/* NOTE: This transposes the output triples. This should ensure they're sorted by column,
 * which is necessary for the multiway merge routine to function
 */
template <typename IT, typename IT2, typename DT>
__global__ void dCSR_to_triples(DT * d_vals, IT * d_colinds, IT * d_rowptrs, 
                                std::tuple<IT, IT, DT> * d_triples,
                                const IT2 rows, const IT offset=0)
{
    const uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    const uint32_t wid = tid / warpSize;
    const uint32_t lid = tid % warpSize;

    if (wid < rows) {
        IT start = d_rowptrs[wid];
        IT end = d_rowptrs[wid+1];
        for (int j = start + lid; j < end; j += warpSize)
        {
            std::get<0>(d_triples[j]) = d_colinds[j];
            std::get<1>(d_triples[j]) = wid + offset;
            std::get<2>(d_triples[j]) = d_vals[j];
        }
    } 
}


/* Call GALATIC SpGEMM, convert output to tuples, return pointer to tuples on device */
template <typename SR, typename IT, typename DT>
std::tuple<IT, IT, DT> * local_spgemm_galatic(dCSR<DT>& A, dCSR<DT>& A_t, 
                                                IT& nnz,IT offset = 0)
{
    assert((A.nnz > 0) && (A_t.nnz > 0));

    using Triple = std::tuple<IT, IT, DT>;

    dCSR<DT> C;
    SR semiring;

    const IT m_threshold = 100; //GALATIC explodes if matrices are small
    if (A.rows < m_threshold || A_t.rows < m_threshold)
    {
        //CPU multiply for now
        CSR<DT> h_A;
        CSR<DT> h_A_t;
        CSR<DT> h_C_csr;
        h_A.alloc(A.rows, A.cols, A.nnz);
        h_A_t.alloc(A_t.rows, A_t.cols, A_t.nnz);

        convert(h_A, A);
        convert(h_A_t, A_t);

        Mult_CPU<SR>(h_A, h_A_t, h_C_csr, semiring);

        convert(C, h_C_csr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    else
    {
        // GPU multiply

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

        /* Do multiply */
        ACSpGEMM::Multiply<SR>(A, A_t, C, DefaultTraits, stats, false, semiring);
    }

    nnz = C.nnz;

    //dump_dCSR_to_log(logptr, C);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Convert to device triples */
    Triple * d_triples;
    CUDA_CHECK(cudaMalloc(&d_triples, sizeof(Triple)*nnz));

    const uint32_t tpb = 256;
    const uint32_t wpb = std::min(C.rows, static_cast<size_t>(tpb / 32));
    const uint32_t blocks = std::ceil( static_cast<double>(C.rows) / static_cast<double>(wpb) );
    dCSR_to_triples<<<blocks, tpb>>>(C.data, C.col_ids, C.row_offsets, d_triples, C.rows, offset);
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_triples;
}


template <typename SR, typename IT, typename DT>
std::tuple<IT, IT, DT> * local_spgemm_galatic(SpMat<IT, DT>& A, SpMat<IT, DT>& B, 
                                                IT& nnz, IT offset = 0)
{
    auto A_dcsr = make_dCSR_from_spmat(A);
    auto B_dcsr = make_dCSR_from_spmat(B);

    /* This is horrible, but no other option for now */
    auto B_t_dcsr = transpose_outofplace<DT>(B_dcsr);
    auto d_triples = local_spgemm_galatic<SR>(A_dcsr, B_t_dcsr, nnz, offset);

    clear_dCSR_ptrs(A_dcsr);
    clear_dCSR_ptrs(B_dcsr);

    return d_triples;
}

template <typename IT, typename DT>
struct is_lower_triangular
{
    __host__ __device__
    bool operator()(std::tuple<IT, IT, DT> t)
    {
        //return std::get<0>(t) > std::get<1>(t);
        return false;
    }
};

template <typename IT, typename DT>
std::tuple<IT, IT, DT> * copy_lower_triangular(std::tuple<IT, IT, DT> * d_triples, const IT n)
{
    thrust::device_ptr<std::tuple<IT, IT, DT>> d_triples_ptr = thrust::device_pointer_cast(d_triples);

    IT nnz_lower = static_cast<IT>(thrust::count_if(d_triples_ptr, d_triples_ptr + n, is_lower_triangular<IT, DT>()));

    thrust::host_vector<std::tuple<IT, IT, DT>> h_triples(nnz_lower);

    thrust::copy_if(d_triples_ptr, d_triples_ptr + n, h_triples.begin(), is_lower_triangular<IT, DT>());

    return thrust::raw_pointer_cast(h_triples.data());

}



}

#endif
