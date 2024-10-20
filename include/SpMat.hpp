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
        this->ds_vals = nvshmem_malloc(nnz * sizeof(DT));
        this->ds_colinds = nvshmem_malloc(nnz * sizeof(IT));
        this->ds_rowptrs = nvshmem_malloc( (m + 1) * sizeof(IT));
    }


    //TODO: Constructor with cootriples


    ~SpMat()
    {
        NVSHMEM_FREE_SAFE(ds_vals);
        NVSHMEM_FREE_SAFE(ds_colinds);
        NVSHMEM_FREE_SAFE(ds_rowptrs);
    }

private:
    IT m, n, nnz;
    DT * ds_vals;
    IT * ds_colinds;
    IT * ds_rowptrs;
};

}

#endif
