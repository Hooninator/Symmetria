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


    void set_from_coo(CooTriples<IT, DT>* triples, bool rowsorted=false)
    {

        assert(m > 0 && loc_m > 0);

        this->loc_nnz = triples->get_nnz();

        /* Allgather to get global tile sizes array */
        IT send = this->loc_nnz;
        MPI_Allgather(&(send), 1, MPIType<IT>(), 
                        this->tile_sizes.data(), 1, MPIType<IT>(), 
                        this->proc_map->get_world_comm());

#ifdef DEBUG
        logptr->log_vec(this->tile_sizes, "Tile sizes");
#endif
        
        this->ds_vals = ((DT *)nvshmem_malloc(this->loc_nnz * sizeof(DT)));

        this->ds_colinds = ((IT *)nvshmem_malloc(this->loc_nnz * sizeof(IT)));
    
        this->ds_rowptrs = ((IT *)nvshmem_malloc((this->loc_m+1) * sizeof(IT)));

        if (this->loc_nnz==0)
            return;

        std::vector<DT> * h_vals = new std::vector<DT>();
        h_vals->reserve(this->loc_nnz);

        std::vector<IT> * h_colinds = new std::vector<IT>();
        h_colinds->reserve(this->loc_nnz);

        std::vector<IT> * h_rowptrs = new std::vector<IT>();
        h_rowptrs->reserve(this->loc_m + 1);

        // This is essential, otherwise setting rowptrs array is difficult
        
        START_TIMER("RowSort");
        if (!rowsorted)
            triples->rowsort();
        STOP_TIMER("RowSort");

        START_TIMER("CSRConstruction");

        IT prev_row = std::get<0>(triples->at(0));

        for (IT j=0; j<=prev_row; j++) {
            h_rowptrs->emplace_back(0);
        }

        for (IT i=0; i<triples->size(); i++) {
            IT row = std::get<0>(triples->at(i));

            if (row != prev_row) {

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

        STOP_TIMER("CSRConstruction");

#ifdef DEBUG
        logptr->log_vec(*h_rowptrs, "rowptrs", "End rowptrs");
#endif

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

        MPI_Barrier(this->proc_map->get_world_comm());
    }


    ~DistSpMat()
    {
        NVSHMEM_FREE_SAFE(ds_vals);
        NVSHMEM_FREE_SAFE(ds_colinds);
        NVSHMEM_FREE_SAFE(ds_rowptrs);
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


    Triple map_glob_to_local(Triple& t)
    {
        IT row = std::get<0>(t);
        IT col = std::get<1>(t);
        DT val = std::get<2>(t);

        return {row % this->loc_m, col, val};
    }


    int map_triple(Triple& t)
    {
        return std::get<0>(t) / this->loc_m;
    }


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
}
#endif
