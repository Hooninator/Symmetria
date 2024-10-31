#ifndef SP_MAT_HPP
#define SP_MAT_HPP

#include "common.h"
#include "CooTriples.hpp"
#include "dCSR.cuh"
#include "dCSR_utils.cuh"
#include "kernels/make_csr.cuh"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>


namespace symmetria
{

template <typename IT, typename DT>
class SpMat
{
    using Triple = std::tuple<IT, IT, DT>;

public:

    SpMat(const IT m, const IT n, CooTriples<IT, DT>& triples):
        m(m), n(n), nnz(triples.get_nnz())
    {
        //TODO: store in contiguous buffer from one malloc
        CUDA_CHECK(cudaMalloc(&this->ds_vals, nnz * sizeof(DT)));
        CUDA_CHECK(cudaMalloc(&this->ds_colinds, nnz * sizeof(IT)));
        CUDA_CHECK(cudaMalloc(&this->ds_rowptrs, (m + 1) * sizeof(IT)));

        //TODO: Make MPI_Wins referencing each buffer


        build_csr_fast(triples, m);
    }


    void build_csr_fast(CooTriples<IT, DT>& triples,
                        const IT rows)
    {
        thrust::host_vector<IT> row_nnz(rows);
        thrust::for_each(triples.begin(), triples.end(),
            [&](auto const& t)mutable
            {
                row_nnz[std::get<0>(t)]++;
            }
        );

        thrust::device_vector<IT> d_row_nnz(row_nnz.begin(), row_nnz.end());
        thrust::inclusive_scan(d_row_nnz.begin(), d_row_nnz.end(), 
                                thrust::device_pointer_cast(this->ds_rowptrs)+1);

        thrust::fill_n(d_row_nnz.begin(), rows, IT(0));

        Triple * d_triples;
        CUDA_CHECK(cudaMalloc(&d_triples, sizeof(Triple) * triples.get_nnz()));
        CUDA_CHECK(cudaMemcpy(d_triples, triples.get_triples().data(), sizeof(Triple) * triples.get_nnz(),
                                cudaMemcpyHostToDevice));

        const uint32_t tpb = 512;
        const uint32_t blocks = std::ceil( static_cast<double>(triples.get_nnz()) / static_cast<double>(tpb) );
        make_csr<<<blocks, tpb>>>(d_triples,
                                    thrust::raw_pointer_cast(d_row_nnz.data()),
                                    this->ds_vals, this->ds_colinds, this->ds_rowptrs,
                                    triples.get_nnz());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_FREE_SAFE(d_triples);
    }


    inline IT get_m() {return m;}
    inline IT get_n() {return n;}
    inline IT get_nnz() {return nnz;}


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
