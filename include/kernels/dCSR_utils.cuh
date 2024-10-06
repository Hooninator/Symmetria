#ifndef DCSR_UTILS_CUH
#define DCSR_UTILS_CUH

#include "../common.h"

#include "dCSR.cuh"
#include "SemiRingInterface.h"
#include "../source/device/Multiply.cuh"

#include <thrust/device_ptr.h>
#include <thrust/mismatch.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>


namespace symmetria {

template <typename T, typename Mat, typename I=unsigned int>
void make_dCSR_from_distspmat(Mat& A_dist, dCSR<T>& A)
{
    A.rows = A_dist.get_loc_rows();
    A.cols = A_dist.get_loc_cols();
    A.nnz = A_dist.get_loc_nnz();
    A.data = A_dist.get_vals();
    A.col_ids = (I*)A_dist.get_colinds();
    A.row_offsets = (I*)A_dist.get_rowptrs();
}


template <typename T, typename Mat, typename I=unsigned int>
dCSR<T> make_dCSR_from_distspmat_outofplace(Mat& A_dist)
{
    dCSR<T> A;
    A.rows = A_dist.get_loc_rows();
    A.cols = A_dist.get_loc_cols();
    A.nnz = A_dist.get_loc_nnz();

    A.alloc(A.rows, A.cols, A.nnz);

    CUDA_CHECK(cudaMemcpy(A.data, A_dist.get_vals(), sizeof(T) * A.nnz,
                            cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(A.col_ids, A_dist.get_colinds(), sizeof(I) * A.nnz,
                            cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(A.row_offsets, A_dist.get_rowptrs(), sizeof(I) * (A.rows + 1),
                            cudaMemcpyDeviceToDevice));

    return A;
}


template <typename T>
void dump_dCSR_to_log(Log * logfile, dCSR<T>& A)
{
    logfile->log_device_array(A.data, A.nnz, "Values:");
    logfile->log_device_array(A.col_ids, A.nnz, "Colinds:");
    logfile->log_device_array(A.row_offsets, A.rows+1, "Rowptrs:");
}



template <typename T>
bool operator==(const dCSR<T>& lhs, const dCSR<T>& rhs)
{

    double eps = 1e-3;

    /* Dimensions and nonzeros */
    if (lhs.nnz != rhs.nnz ||
        lhs.rows != rhs.rows ||
        lhs.cols != rhs.cols) {
        return false;
    }


    /* Make sure nonzeros are close */

    thrust::device_ptr<T> lhs_vals = thrust::device_pointer_cast(lhs.data);
    thrust::device_ptr<T> rhs_vals = thrust::device_pointer_cast(rhs.data);

    thrust::device_vector<T> temp(lhs.nnz);

    thrust::transform(lhs_vals, lhs_vals + lhs.nnz, rhs_vals, temp.begin(), thrust::minus<T>());
    T err = thrust::transform_reduce(temp.begin(), temp.end(), thrust::square<T>(), 0.0, thrust::plus<T>());

    return err < eps; 

}
    

}

#endif
