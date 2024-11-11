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

    SpMat(): m(0), n(0), nnz(0), total_bytes(0), baseptr(nullptr),
            ds_vals(nullptr), ds_colinds(nullptr), ds_rowptrs(nullptr){}

    SpMat(const IT m, const IT n, CooTriples<IT, DT>& triples,
            char * baseptr, const bool transpose):
        m(m), n(n), nnz(triples.get_nnz()), baseptr(baseptr)
    {
        assert(sizeof(DT) >= sizeof(IT)); //otherwise alignment doesn't work

        if (transpose)
            std::swap(this->m, this->n);

		size_t offset_colinds = aligned_offset<DT>(nnz * sizeof(DT));  
		size_t offset_rowptrs = aligned_offset<DT>(offset_colinds + nnz * sizeof(IT)) ;

        if (nnz==0) {
            total_bytes = 0;
        } else {

            total_bytes = aligned_offset<DT>(offset_rowptrs + (m+ 1) * sizeof(IT));

            this->ds_vals = (DT*)this->baseptr;
            this->ds_colinds = (IT*)(this->baseptr + offset_colinds);
            this->ds_rowptrs = (IT*)(this->baseptr + offset_rowptrs);

            if (transpose) {
                build_csr<1, 0>(triples, m);
            } else {
                build_csr<0, 1>(triples, m);
            }

        }
    }


    SpMat(const IT m, const IT n, const IT nnz, char * d_buffer):
        m(m), n(n), nnz(nnz), baseptr(d_buffer)
    {
        assert(nnz>0);
        assert(sizeof(DT) >= sizeof(IT)); //otherwise alignment doesn't work

		size_t offset_colinds = aligned_offset<DT>(nnz * sizeof(DT));  
		size_t offset_rowptrs = aligned_offset<DT>(offset_colinds + nnz * sizeof(IT)) ;

        total_bytes = aligned_offset<DT>(offset_rowptrs + (m + 1) * sizeof(IT));

        this->ds_vals = (DT*)d_buffer;
        this->ds_colinds = (IT*)(d_buffer + offset_colinds);
        this->ds_rowptrs = (IT*)(d_buffer + offset_rowptrs);
    }


    template <int RI, int CI>
    void build_csr(CooTriples<IT, DT>& triples, const IT rows)
    {
        thrust::host_vector<IT> row_nnz(rows);
        thrust::for_each(triples.begin(), triples.end(),
            [&](auto const& t)mutable
            {
                row_nnz[std::get<RI>(t)]++;
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
        make_csr<IT, DT, RI, CI><<<blocks, tpb>>>(d_triples,
                                        thrust::raw_pointer_cast(d_row_nnz.data()),
                                        this->ds_vals, this->ds_colinds, this->ds_rowptrs,
                                        triples.get_nnz());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_FREE_SAFE(d_triples);
    }


    void dump_to_log(Log * logfile, const char * prefix)
    {
        logfile->OFS()<<prefix<<std::endl;
        logfile->OFS()<<"nnz: "<<this->nnz<<", m: "<<this->m<<", n: "<<this->n<<std::endl;

        if (nnz==0) return;

        std::vector<DT> h_vals(this->nnz);
        std::vector<IT> h_colinds(this->nnz);
        std::vector<IT> h_rowptrs(this->m + 1);

        CUDA_CHECK(cudaMemcpy(h_vals.data(), ds_vals, sizeof(DT)*this->nnz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_colinds.data(), ds_colinds, sizeof(IT)*this->nnz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rowptrs.data(), ds_rowptrs, sizeof(IT)*(this->m + 1), cudaMemcpyDeviceToHost));

        CooTriples<IT, DT> triples(&h_vals, &h_colinds, &h_rowptrs);
        triples.dump_to_log(logfile);
    }


    inline IT get_m() const {return m;}
    inline IT get_n() const {return n;}
    inline IT get_nnz() const {return nnz;}
    inline uint64_t get_total_bytes() {return total_bytes;}

    inline DT * get_vals() const {return ds_vals;}
    inline IT * get_colinds() const {return ds_colinds;}
    inline IT * get_rowptrs() const {return ds_rowptrs;}

    template< typename IT2, typename DT2>
    friend bool operator==(const SpMat<IT2, DT2>& lhs, const SpMat<IT2, DT2>& rhs);

    void free()
    {
        CUDA_FREE_SAFE(baseptr);
    }

    ~SpMat()
    {
    }

private:
    IT m, n, nnz;
    DT * ds_vals;
    IT * ds_colinds;
    IT * ds_rowptrs;

    char * baseptr;
    uint64_t total_bytes;
};


/* NOTE: This is inefficient */
template <typename IT, typename DT>
bool operator==(const SpMat<IT, DT>& lhs, const SpMat<IT, DT>& rhs)
{
    DT * h_lhs_vals = new DT[lhs.nnz];
    DT * h_rhs_vals = new DT[rhs.nnz];
    IT * h_lhs_colinds = new IT[lhs.nnz];
    IT * h_rhs_colinds = new IT[rhs.nnz];
    IT * h_lhs_rowptrs = new IT[lhs.m + 1];
    IT * h_rhs_rowptrs = new IT[rhs.m + 1];

    CUDA_CHECK(cudaMemcpy(h_lhs_vals, lhs.ds_vals, sizeof(DT)*lhs.nnz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rhs_vals, rhs.ds_vals, sizeof(DT)*rhs.nnz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lhs_colinds, lhs.ds_colinds, sizeof(IT)*lhs.nnz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rhs_colinds, rhs.ds_colinds, sizeof(IT)*rhs.nnz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lhs_rowptrs, lhs.ds_rowptrs, sizeof(IT)*(lhs.m+1), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rhs_rowptrs, rhs.ds_rowptrs, sizeof(IT)*(rhs.m+1), cudaMemcpyDeviceToHost));

    CooTriples<IT, DT> lhs_triples(h_lhs_vals, h_lhs_colinds, h_lhs_rowptrs, lhs.nnz, lhs.m);
    CooTriples<IT, DT> rhs_triples(h_rhs_vals, h_rhs_colinds, h_rhs_rowptrs, rhs.nnz, rhs.m);

#ifdef DEBUG
    lhs_triples.dump_to_log(logptr, "LHS");
    rhs_triples.dump_to_log(logptr, "RHS");
#endif

    /* Dimensions and nnz */
    if (lhs.nnz != rhs.nnz ||
        lhs.m!= rhs.m||
        lhs.m!= rhs.m) {
        return false;
    }


    bool correct = (lhs_triples == rhs_triples);

    delete[] h_lhs_vals;
    delete[] h_rhs_vals;
    delete[] h_lhs_colinds;
    delete[] h_rhs_colinds;
    delete[] h_lhs_rowptrs;
    delete[] h_rhs_rowptrs;

    return correct;
}




}

#endif
