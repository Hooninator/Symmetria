#ifndef DIST_SP_MAT_HPP
#define DIST_SP_MAT_HPP

#include "common.h"
#include "ProcMap.hpp"
#include "CooTriples.hpp"

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
    }


    void set_from_coo(CooTriples<IT, DT>* triples)
    {

        assert(m > 0 && loc_m > 0);

        this->loc_nnz = triples->get_nnz();

        DEBUG_PRINT_ALL("Local nnz " + STR(this->loc_nnz));
        DEBUG_PRINT_ALL("Local rows " + STR(this->loc_m));

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
    
        this->ds_rowptrs = ((IT *)nvshmem_malloc((this->loc_m+1) * sizeof(DT)));

        if (this->loc_nnz==0)
            return;

        std::vector<DT> * h_vals = new std::vector<DT>();
        h_vals->reserve(this->loc_nnz);

        std::vector<IT> * h_colinds = new std::vector<IT>();
        h_colinds->reserve(this->loc_nnz);

        std::vector<IT> * h_rowptrs = new std::vector<IT>();
        h_rowptrs->reserve(this->loc_m + 1);

        // This is essential, otherwise setting rowptrs array is difficult
        triples->rowsort();

        //TODO: Use std::generate for this
        IT prev_row = std::get<0>(triples->at(0));
        for (IT j=0; j<=prev_row; j++) {
            h_rowptrs->emplace_back(0);
        }

        for (IT i=0; i<triples->size(); i++) {
            IT row = std::get<1>(triples->at(i));

            if (row != prev_row) {

                for (IT j=0; j<(row-prev_row); j++) {
                    h_rowptrs->emplace_back(i);
                }

                prev_row = row;
            }

            h_vals->emplace_back(std::get<2>(triples->at(i)));
            h_colinds->emplace_back(std::get<1>(triples->at(i)));

        }
        h_rowptrs->emplace_back(triples->size());

#ifdef DEBUG_DIST_SPMAT
        //logptr->print_vec(*h_rowptrs, "rowptrs", "End rowptrs");
#endif
            
        CUDA_CHECK(cudaMemcpyAsync(this->ds_vals, h_vals->data(), h_vals->size()*sizeof(DT),
                                cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(this->ds_colinds, h_colinds->data(), h_colinds->size()*sizeof(IT),
                                cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(this->ds_rowptrs, h_rowptrs->data(), h_rowptrs->size()*sizeof(IT),
                                cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaDeviceSynchronize());

        delete h_vals;
        delete h_colinds;
        delete h_rowptrs;
    }


    /* Non blocking broadcast of tile of matrix that lives on root.
     * Results are stored in the vectors passed into the function.
     */
    void ibcast_tile(const int root, 
                     std::vector<DT>& vals,
                     std::vector<IT>& colinds,
                     std::vector<IT>& rowptrs,
                     std::vector<MPI_Request>& requests)
    {

        MPI_Ibcast(vals.data(), vals.size(), MPIType<DT>(),
                   root, proc_map->get_world_comm(),
                   &requests[0]);

        MPI_Ibcast(colinds.data(), vals.size(), MPIType<IT>(),
                   root, proc_map->get_world_comm(),
                   &requests[1]);

        MPI_Ibcast(rowptrs.data(), vals.size(), MPIType<IT>(),
                   root, proc_map->get_world_comm(),
                   &requests[2]);

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

    inline IT get_loc_rows() const {return loc_m;}
    inline IT get_loc_cols() const {return loc_n;}
    inline IT get_loc_nnz() const {return loc_nnz;}

    inline DT * get_vals() const {return ds_vals;}
    inline IT * get_colinds() const {return ds_colinds;}
    inline IT * get_rowptrs() const {return ds_rowptrs;}

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

}
#endif
