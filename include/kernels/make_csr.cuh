#ifndef MAKE_CSR_CUH
#define MAKE_CSR_CUH

#include "../common.h"


namespace symmetria {

template <typename IT, typename DT, int RI, int CI>
__global__ void make_csr(std::tuple<IT, IT, DT> * d_triples,
                         IT * row_nnz,
                         DT * d_vals,
                         IT * d_colinds,
                         const IT * d_rowptrs,
                         const IT nnz)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nnz)
    {
        const IT rid = std::get<RI>(d_triples[tid]);
        const IT idx = d_rowptrs[rid] + atomicAdd(row_nnz + rid, 1);
        d_colinds[idx] = std::get<CI>(d_triples[tid]);
        d_vals[idx] = std::get<2>(d_triples[tid]);
    }
}



}



#endif
