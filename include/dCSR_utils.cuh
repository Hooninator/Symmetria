#ifndef DCSR_UTILS_CUH
#define DCSR_UTILS_CUH

#include "common.h"
#include "SpMat.hpp"

#include "dCSR.cuh"
#include "CSR.cuh"
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
dCSR<DT> make_dCSR_from_spmat(const SpMat<IT, DT>& A)
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
dCSR<DT> make_dCSR_from_spmat_outofplace(const SpMat<IT, DT>& A)
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
    logfile->OFS()<<"nnz: "<<A.nnz<<", m: "<<A.rows<<", n: "<<A.cols<<std::endl;

    if (A.nnz==0) return;

    std::vector<T> h_vals(A.nnz);
    std::vector<unsigned int> h_colinds(A.nnz);
    std::vector<unsigned int> h_rowptrs(A.rows + 1);

    CUDA_CHECK(cudaMemcpy(h_vals.data(), A.data, sizeof(T)*A.nnz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_colinds.data(), A.col_ids, sizeof(unsigned int)*A.nnz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rowptrs.data(), A.row_offsets, sizeof(unsigned int)*(A.rows + 1), cudaMemcpyDeviceToHost));

    CooTriples<unsigned int, T> triples(&h_vals, &h_colinds, &h_rowptrs);
    triples.dump_to_log(logptr);
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

    CSR<T> h_lhs, h_rhs;
    convert(h_lhs, lhs);
    convert(h_rhs, rhs);

    for (int i=0; i<rhs.nnz; i++)
    {
        if (fabs(h_lhs.data[i] - h_rhs.data[i]) > eps)
            return false;
    }
    
    return true;
}
    

}

#endif
