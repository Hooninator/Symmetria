#ifndef DIST_SP_MAT_CYCLIC_HPP
#define DIST_SP_MAT_CYCLIC_HPP

#include "common.h"
#include "ProcMap.hpp"
#include "CooTriples.hpp"
#include "dCSR.cuh"
#include "dCSR_utils.cuh"
#include "SpMat.hpp"

namespace symmetria
{


template <typename IT, typename DT>
class DistSpMatCyclic
{

public:

    DistSpMatCyclic(){}

    DistSpMatCyclic(const IT m, const IT n, const IT nnz,
                    const IT mb, const IT nb, 
                    std::shared_ptr<ProcMap> proc_map):
        m(m), n(n), nnz(nnz),
        mb(mb), nb(nb),
        mtiles(m / mb), ntiles(n / nb),
        proc_map(proc_map)
    {
        assert(mb <= (m / proc_map->get_px()));
        assert(nb <= (n / proc_map->get_py()));
    }


    DistSpMatCyclic(const IT m, const IT n, const IT nnz,
                    const IT mb, const IT nb,
                    std::shared_ptr<ProcMap> proc_map,
                    std::vector<SpMat<IT, DT>>& local_matrices):
        m(m), n(n), nnz(nnz),
        mtiles(m / mb), ntiles(n / nb),
        local_matrices(local_matrices),
        proc_map(proc_map)
    {
        assert(mb <= (m / proc_map->get_px()));
        assert(nb <= (n / proc_map->get_py()));
    }


    void set_from_coo(CooTriples<IT, DT> * triples)
    {

        /* Map each triple to the tile that owns it */

        /* Convert indices to tile indices */

        /* Build the CSR arrays for each tile */

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


    inline void set_rows(const IT _m) {m=_m;}
    inline void set_cols(const IT _n) {n=_n;}
    inline void set_nnz(const IT _nnz) {nnz=_nnz;}

    inline IT get_rows() {return m;}
    inline IT get_cols() {return n;}
    inline IT get_nnz() {return nnz;}

    inline std::vector<SpMat<IT, DT>> get_local_matrices() {return local_matrices;}

    std::shared_ptr<ProcMap> proc_map;

protected:
    IT m, n, nnz;
    std::vector<IT> tile_sizes;

    IT mb, nb;
    IT mtiles, ntiles;
    std::vector<SpMat<IT, DT>> local_matrices;

};


template <typename IT, typename DT>
class DistSpMatCyclic2D : public DistSpMatCyclic<IT, DT>
{
public:

    using Triple = std::tuple<IT, IT, DT>;

    DistSpMatCyclic2D(const IT m, const IT n, const IT nnz,
                    const IT mb, const IT nb, 
                    std::shared_ptr<ProcMap> proc_map):
        DistSpMatCyclic<IT, DT>(m, n, nnz, mb, nb, proc_map)
    {
    }


    Triple map_glob_to_local(const Triple& t)
    {
        return t;
    }


    /* This maps a triple with global coordinates to local tile coordinates */
    Triple map_glob_to_tile(const Triple& t)
    {

        IT loc_i = std::get<0>(t) % this->mb;
        IT loc_j = std::get<1>(t) % this->nb;

        IT mp = (this->m / this->mb) * this->mb;
        IT np = (this->n / this->nb) * this->nb;

        if (std::get<0>(t) >= this->mp)
            loc_i += (std::get<t>(0) - this->mp);
        if (std::get<1>(t) >= this->np)
            loc_j += (std::get<t>(1) - this->np);
        
        return {loc_i, loc_j, std::get<2>(t)};

    }


    /* Which process owns this */
    int owner(const Triple& t)
    {
        IT i = std::get<0>(t);
        IT j = std::get<1>(t);

        int row_contrib = (std::min((i / this->mb), this->mtiles - 1) % this->proc_map->get_px())
                            * this->proc_map->get_py();
        int col_contrib = std::min((j / this->nb), this->ntiles - 1) % this->proc_map->get_py();

        assert ((row_contrib + col_contrib) < this->proc_map->get_grid_size());

        return row_contrib + col_contrib;
    }


    /* Which of my tiles owns this.
     * The tiles are stored in row major order 
     * At this point, we can assume the triple is local to the process
     * but its indices are global.
     */
    int tile_owner(const Triple& t)
    {
        IT i = std::get<0>(t);
        IT j = std::get<1>(t);
        //int num_tiles = (this->mtiles / this->proc_map->get_px()) * (this->ntiles / this->proc_map->get_py());
        int row_contrib = std::min(i / (this->mb * this->proc_map->get_px()), 
                                  this->mtiles / this->proc_map->get_px()) 
                            * (this->ntiles / this->proc_map->get_py());
        int col_contrib = std::min(j / (this->nb * this->proc_map->get_py()), 
                                    this->ntiles / this->proc_map->get_py());
        return row_contrib + col_contrib;
    }


};

}






#endif
