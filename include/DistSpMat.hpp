#ifndef DIST_SP_MAT_HPP
#define DIST_SP_MAT_HPP

#include "common.h"
#include "ProcMap.hpp"
#include "CooTriples.hpp"
#include "dCSR.cuh"
#include "dCSR_utils.cuh"

namespace symmetria {


/* Base class for a distributed sparse matrix.
 * Derived classes correspond to specific distributions e.g. 2D, 1D block row, etc.
 */
template <typename IT, typename DT>
class DistSpMat
{

public:

    DistSpMat(){}

    DistSpMat(const IT m, const IT n, const IT nnz,
              std::shared_ptr<ProcMap> proc_map):
        m(m), n(n), nnz(nnz), proc_map(proc_map),
        tile_sizes(proc_map->get_grid_size())
    {
        MPI_Barrier(proc_map->get_world_comm());
    }


    void set_from_coo(CooTriples<IT, DT>* triples)
    {

        assert(this->m > 0 && this->loc_m > 0);

        this->loc_nnz = triples->get_nnz();

        /* Allgather to get global tile sizes array */
        IT send = this->loc_nnz;
        MPI_Allgather(&(send), 1, MPIType<IT>(), 
                        this->tile_sizes.data(), 1, MPIType<IT>(), 
                        this->proc_map->get_world_comm());

        CUDA_CHECK(cudaMalloc(&this->ds_vals, this->loc_nnz * sizeof(DT)));
        CUDA_CHECK(cudaMalloc(&this->ds_colinds, this->loc_nnz * sizeof(IT)));
        CUDA_CHECK(cudaMalloc(&this->ds_rowptrs, (this->loc_m + 1) * sizeof(IT)));

        if (this->loc_nnz==0)
            return;
        
        START_TIMER("CSRConstruction");

        std::vector<DT> * h_vals = new std::vector<DT>();
        h_vals->resize(this->loc_nnz);

        std::vector<IT> * h_colinds = new std::vector<IT>();
        h_colinds->resize(this->loc_nnz);

        std::vector<IT> * h_rowptrs = new std::vector<IT>();
        h_rowptrs->resize(this->loc_m + 1);

        this->build_csr_fast(triples, h_vals, h_colinds, h_rowptrs, this->loc_m);

        STOP_TIMER("CSRConstruction");

        START_TIMER("CSRCopy");
            
        CUDA_CHECK(cudaMemcpyAsync(this->ds_vals, h_vals->data(), h_vals->size()*sizeof(DT),
                                cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(this->ds_colinds, h_colinds->data(), h_colinds->size()*sizeof(IT),
                                cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(this->ds_rowptrs, h_rowptrs->data(), h_rowptrs->size()*sizeof(IT),
                                cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        STOP_TIMER("CSRCopy");

        delete h_vals;
        delete h_colinds;
        delete h_rowptrs;
        delete triples;

        MPI_Barrier(this->proc_map->get_world_comm());
    }


    void build_csr_fast(CooTriples<IT, DT> * triples,
                        std::vector<DT> * h_vals,
                        std::vector<IT> * h_colinds,
                        std::vector<IT> * h_rowptrs,
                        const IT rows)
    {
        std::vector<IT> row_nnz(rows);
        std::for_each(triples->begin(), triples->end(),
            [&](auto const& t)mutable
            {
                row_nnz[std::get<0>(t)]++;
            }
        );

        std::inclusive_scan(row_nnz.begin(), row_nnz.end(), 
                            h_rowptrs->begin()+1);

        std::fill(row_nnz.begin(), row_nnz.end(), IT(0));

        for (int i=0; i<triples->get_nnz(); i++)
        {
            auto const t = triples->at(i);
            const IT rid = std::get<0>(t);
            const IT idx = h_rowptrs->at(rid) + row_nnz[rid];
            h_colinds->at(idx) = std::get<1>(t);
            h_vals->at(idx) = std::get<2>(t);
            row_nnz[rid]++;
        }

    }


    void build_csr(CooTriples<IT, DT> * triples,
                    std::vector<DT> * h_vals,
                    std::vector<IT> * h_colinds,
                    std::vector<IT> * h_rowptrs)
    {
        IT prev_row = std::get<0>(triples->at(0));

        for (IT j=0; j<=prev_row; j++) {
            h_rowptrs->emplace_back(0);
        }

        for (IT i=0; i<triples->size(); i++) {
            IT row = std::get<0>(triples->at(i));

            if (row != prev_row) {

                assert( (row - prev_row) >= 0 );

                for (IT j=0; j<(row-prev_row); j++) {
                    h_rowptrs->emplace_back(i);
                }

                prev_row = row;
            }

            h_vals->emplace_back(std::get<2>(triples->at(i)));
            h_colinds->emplace_back(std::get<1>(triples->at(i)));

        }

        while (h_rowptrs->size() < (this->loc_m + 1))
            h_rowptrs->emplace_back(triples->size());

    }


    void build_csr_transpose(CooTriples<IT, DT> * triples,
                            std::vector<DT> * h_vals,
                            std::vector<IT> * h_colinds,
                            std::vector<IT> * h_rowptrs)
    {
        IT prev_row = std::get<1>(triples->at(0));

        for (IT j=0; j<=prev_row; j++) {
            h_rowptrs->emplace_back(0);
        }

        for (IT i=0; i<triples->size(); i++) {
            IT row = std::get<1>(triples->at(i));

            if (row != prev_row) {

                assert( (row - prev_row) >= 0);

                for (IT j=0; j<(row-prev_row); j++) {
                    h_rowptrs->emplace_back(i);
                }

                prev_row = row;
            }

            h_vals->emplace_back(std::get<2>(triples->at(i)));
            h_colinds->emplace_back(std::get<0>(triples->at(i)));

        }

        while (h_rowptrs->size() < (this->loc_m + 1))
            h_rowptrs->emplace_back(triples->size());

        DEBUG_PRINT("Ending while loop");

    }


    ~DistSpMat()
    {
        CUDA_FREE_SAFE(ds_vals);
        CUDA_FREE_SAFE(ds_colinds);
        CUDA_FREE_SAFE(ds_rowptrs);
    }

    inline void set_rows(const IT _m) {m=_m;}
    inline void set_cols(const IT _n) {n=_n;}
    inline void set_nnz(const IT _nnz) {nnz=_nnz;}

    inline IT get_rows() {return m;}
    inline IT get_cols() {return n;}
    inline IT get_nnz() {return nnz;}

    inline IT get_loc_rows() const {return loc_m;}
    inline IT get_loc_cols() const {return loc_n;}
    inline IT get_loc_nnz() const {return loc_nnz;}

    inline DT * get_vals() const {return ds_vals;}
    inline IT * get_colinds() const {return ds_colinds;}
    inline IT * get_rowptrs() const {return ds_rowptrs;}

    inline std::vector<IT> get_tile_sizes() const {return tile_sizes;}

    template <typename IT2, typename DT2>
    friend bool operator==(DistSpMat<IT2, DT2>& lhs, DistSpMat<IT2, DT2>& rhs);

    std::shared_ptr<ProcMap> proc_map;

protected:
    IT m, n, nnz;

    IT loc_m, loc_n, loc_nnz;

    DT * ds_vals;
    IT * ds_colinds, * ds_rowptrs;

    std::vector<IT> tile_sizes;

};


template <typename IT, typename DT>
bool operator==(DistSpMat<IT, DT>& lhs, DistSpMat<IT, DT>& rhs) 
{
    auto A_lhs = make_dCSR_from_distspmat<DT>(lhs);
    auto A_rhs = make_dCSR_from_distspmat<DT>(rhs);

    bool equals = A_lhs == A_rhs;

    MPI_Allreduce(MPI_IN_PLACE, &equals, 1, MPI_INT, MPI_LAND, lhs.proc_map->get_world_comm());

    clear_dCSR_ptrs(A_lhs);
    clear_dCSR_ptrs(A_rhs);

    return equals;
}


template <typename IT, typename DT>
class DistSpMat1DBlockRow : public DistSpMat<IT, DT>
{

    template <typename IU, typename DU>
    friend DistSpMat1DBlockRow spsyrk_bulksync_1d_rowblock(DistSpMat1DBlockRow<IT, DT>& A);


    using Triple = std::tuple<IT, IT, DT>;

public:


    DistSpMat1DBlockRow(const IT m, const IT n, const IT nnz,
                        std::shared_ptr<ProcMap> proc_map):
        DistSpMat<IT, DT>(m, n, nnz, proc_map)
    {
        set_loc_dims();
    }


    void set_loc_dims()
    {
        this->loc_m = std::ceil(static_cast<double>(this->m) / this->proc_map->get_px());
        this->loc_n = this->n;
    }


    Triple map_glob_to_local(const Triple& t)
    {
        IT row = std::get<0>(t);
        IT col = std::get<1>(t);
        DT val = std::get<2>(t);

        return {row % this->loc_m, col, val};
    }


    int owner(const Triple& t)
    {
        return std::get<0>(t) / this->loc_m;
    }


};


}
#endif
