#ifndef TRANSPOSE_CSR_CUH
#define TRANSPOSE_CSR_CUH

#include "../common.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/counting_iterator.h>

namespace symmetria {


/* Count nnz per column, return result in a device vector.
 * Approach is the same as https://github.com/NVIDIA/thrust/blob/master/examples/histogram.cu */
template <typename IT1, typename IT2>
thrust::device_vector<IT1> count_nnz_per_col(const IT2 * d_colinds, 
                                                const IT1 nnz, const IT1 ncols)
{

    thrust::device_vector<IT1> histogram(ncols);
    thrust::device_vector<IT2> colinds_vec(d_colinds, d_colinds + nnz);

    thrust::sort(colinds_vec.begin(), colinds_vec.end());
    thrust::counting_iterator<IT1> search(0);
    thrust::upper_bound(colinds_vec.begin(), colinds_vec.end(),
                        search, search + ncols,
                        histogram.begin());

    thrust::adjacent_difference(histogram.begin(), histogram.end(), histogram.begin());

    return histogram;

}


/* Perform the actual transpose. One warp per row */
template <typename DT, typename IT>
__global__ void transpose_kernel(const DT * d_vals, const IT * d_colinds, const IT * d_rowptrs,
                                DT * d_vals_tr, IT * d_colinds_tr, IT * d_rowptrs_tr, unsigned int * d_offsets,
                                const size_t rows, const size_t cols, const size_t nnz) 
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t wid = tid / warpSize;
    const uint32_t lid = tid % warpSize;

    if (wid < rows) {
        const IT start = d_rowptrs[wid];
        const IT end = d_rowptrs[wid+1];
        for (int l=start+lid; l<end; l += warpSize)
        {
            if (l < end) {
                const IT j = d_colinds[l];
                const DT val = d_vals[l];
                const IT idx_tr = (atomicAdd(d_offsets + j, 1) + d_rowptrs_tr[j]);
                d_vals_tr[idx_tr] = val;
                d_colinds_tr[idx_tr] = wid;
            }
        }
    }

    if (tid==0)
        d_rowptrs_tr[cols] = nnz;
}


/* Out of place transpose of CSR matrix. Returns transposed matrix, input matrix is unaffected.*/
template <typename T>
dCSR<T> transpose_outofplace(const dCSR<T>& A)
{
    auto nnz_per_col = count_nnz_per_col(A.col_ids, A.nnz, A.cols);

    unsigned int * d_offsets;
    CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(unsigned int ) * A.cols));
    CUDA_CHECK(cudaMemset(d_offsets, 0, sizeof(unsigned int ) * A.cols));

    dCSR<T> A_t;
    A_t.alloc(A.cols, A.rows, A.nnz);

    thrust::device_ptr<unsigned int> A_t_rowptrs = thrust::device_pointer_cast(A_t.row_offsets);
    thrust::exclusive_scan(nnz_per_col.begin(), nnz_per_col.end(), A_t_rowptrs);

    const uint32_t warp_size = 32;
    const uint32_t tpb = 512;
    const uint32_t wpb = 512 / warp_size;
    const uint32_t blocks = std::ceil( A.rows / static_cast<double>(wpb));
    transpose_kernel<<<blocks, tpb>>>(A.data, A.col_ids, A.row_offsets,
                                       A_t.data, A_t.col_ids, A_t.row_offsets, 
                                       d_offsets,
                                       A.rows, A.cols, A.nnz);

    CUDA_CHECK(cudaDeviceSynchronize()); 
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaDeviceSynchronize()); 

    return A_t;
}


}




#endif
