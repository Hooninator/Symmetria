#ifndef SP_MAT_HPP
#define SP_MAT_HPP

#include "common.h"
#include "CooTriples.hpp"
#include "dCSR.cuh"
#include "dCSR_utils.cuh"

namespace symmetria
{

template <typename IT, typename DT>
class SpMat
{
    using Triple = std::tuple<IT, IT, DT>;

public:

    SpMat(const IT m, const IT n, const IT nnz):
        m(m), n(n), nnz(nnz)
    {
        CUDA_CHECK(cudaMalloc(&this->ds_vals, nnz * sizeof(DT)));
        CUDA_CHECK(cudaMalloc(&this->ds_colinds, nnz * sizeof(IT)));
        CUDA_CHECK(cudaMalloc(&this->ds_rowptrs, (m + 1) * sizeof(IT)));
    }


    //TODO: Constructor with cootriples


    ~SpMat()
    {
        CUDA_FREE_SAFE(ds_vals);
        CUDA_FREE_SAFE(ds_colinds);
        CUDA_FREE_SAFE(ds_rowptrs);
    }

private:
    IT m, n, nnz;
    DT * ds_vals;
    IT * ds_colinds;
    IT * ds_rowptrs;
};

}

#endif
