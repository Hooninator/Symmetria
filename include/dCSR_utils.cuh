#ifndef DCSR_UTILS_CUH
#define DCSR_UTILS_CUH

#include "common.h"
#include "SpMat.hpp"

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
dCSR<T> make_dCSR_from_distspmat(Mat& A_dist)
{
    dCSR<T> A;
    A.rows = A_dist.get_loc_rows();
    A.cols = A_dist.get_loc_cols();
    A.nnz = A_dist.get_loc_nnz();
    A.data = A_dist.get_vals();
    A.col_ids = A_dist.get_colinds();
    A.row_offsets = A_dist.get_rowptrs();
    return A;
}


template<typename T>
void clear_dCSR_ptrs(dCSR<T>& A)
{
    A.data = nullptr;
    A.col_ids = nullptr;
    A.row_offsets = nullptr;
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


template <typename DT, typename IT>
dCSR<DT> make_dCSR_from_spmat(SpMat<IT, DT>& A)
{
    dCSR<DT> A_dcsr;

    A_dcsr.rows = A.get_m();
    A_dcsr.cols= A.get_n();
    A_dcsr.nnz = A.get_nnz();
    A_dcsr.data = A.get_vals();
    A_dcsr.col_ids = A.get_colinds();
    A_dcsr.row_offsets = A.get_rowptrs();
    
    return A_dcsr;
}


template <typename DT, typename IT>
dCSR<DT> make_dCSR_from_spmat_outofplace(SpMat<IT, DT>& A)
{
    dCSR<DT> A_dcsr;
    A_dcsr.alloc(A.get_m(), A.get_n(), A.get_nnz());

    A_dcsr.data = A.get_vals();
    A_dcsr.col_ids = A.get_colinds();
    A_dcsr.row_offsets = A.get_rowptrs();

    CUDA_CHECK(cudaMemcpy(A_dcsr.data, A.get_vals(), sizeof(DT) * A.get_nnz(),
                            cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(A_dcsr.col_ids, A.get_colinds(), sizeof(IT) * A.get_nnz(),
                            cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(A_dcsr.row_offsets, A.get_rowptrs(), sizeof(IT) * (A.get_m() + 1),
                            cudaMemcpyDeviceToDevice));
    
    return A_dcsr;
}



template <typename T>
void dump_dCSR_to_log(Log * logfile, const dCSR<T>& A)
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
