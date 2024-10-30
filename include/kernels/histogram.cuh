#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

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
thrust::device_vector<IT1> make_histogram(const IT2 * arr, 
                                         const IT1 n, const IT1 nkeys)
{

    thrust::device_vector<IT1> histogram(nkeys);
    thrust::device_vector<IT2> arr_vec(arr, arr+ n);

    thrust::sort(arr_vec.begin(), arr_vec.end());
    thrust::counting_iterator<IT1> search(0);
    thrust::upper_bound(arr_vec.begin(), arr_vec.end(),
                        search, search + nkeys,
                        histogram.begin());

    thrust::adjacent_difference(histogram.begin(), histogram.end(), histogram.begin());

    return histogram;
}



}


#endif
