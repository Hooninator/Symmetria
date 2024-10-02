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


/* Count nnz per column.
 * Approach is the same as https://github.com/NVIDIA/thrust/blob/master/examples/histogram.cu */
template <typename IT>
std::vector<IT> count_nnz_per_col(IT * d_colinds, const IT nnz, const IT ncols)
{

    thrust::device_vector<IT> histogram(ncols);
    thrust::device_vector<IT> colinds_vec(d_colinds, d_colinds + nnz);

    thrust::sort(colinds_vec.begin(), colinds_vec.end());
    thrust::counting_iterator<IT> search(0);
    thrust::upper_bound(colinds_vec.begin(), colinds_vec.end(),
                        search, search + ncols,
                        histogram.begin());

    thrust::adjacent_difference(histogram.begin(), histogram.end(), histogram.begin());

    std::vector<IT> h_histogram(ncols);
    thrust::copy(histogram.begin(), histogram.end(), h_histogram.begin());

}


template <typename IT>
__global__ void tranpose_kernel() {}//TODO



/* Transpose the input matrix */
template <typename T>
void transpose(dCSR<T>& A)
{
    auto nnz_per_col = count_nnz_per_col(A.col_ids, A.nnz, A.cols);
}


}




#endif
