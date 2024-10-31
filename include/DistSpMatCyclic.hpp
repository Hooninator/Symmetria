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


template <typename IT, typename DT, typename P>
class DistSpMatCyclic
{

public:

    using Triple = std::tuple<IT, IT, DT>;

    DistSpMatCyclic(){}

    DistSpMatCyclic(const IT m, const IT n, const IT nnz,
                    const IT mb, const IT nb, 
                    std::shared_ptr<P> proc_map):
        m(m), n(n), nnz(nnz),
        mb(mb), nb(nb),
        mtiles(m / mb), ntiles(n / nb),
        proc_map(proc_map)
    {
        assert(mb <= (m / proc_map->get_px()));
        assert(nb <= (n / proc_map->get_py()));
        tile_sizes.resize(mtiles * ntiles);
    }


    virtual int tile_owner(const Triple& t) {assert(false);}
    virtual Triple map_glob_to_tile(const Triple& t) {assert(false);}


    void set_from_coo(CooTriples<IT, DT> * triples)
    {

        /* Map each triple to the tile that owns it */
        std::vector<CooTriples<IT, DT>> tile_triples(this->n_local_tiles);

        DEBUG_PRINT("Mapping to tiles");

        std::for_each(triples->begin(), triples->end(),
            [&](auto& t)
            {
                int target_tile = this->tile_owner(t);
                tile_triples[target_tile].add_triple(this->map_glob_to_tile(t));
            }
        );


        DEBUG_PRINT("Making CSR arrays");

        /* Build the CSR arrays for each tile */
        for (int i=0; i<tile_triples.size(); i++) 
        {
            auto& this_tile_triples = tile_triples[i];
            local_matrices.emplace_back(mb + row_edge_size(i),
                                          nb + col_edge_size(i), 
                                          this_tile_triples);
        }
        
        DEBUG_PRINT("Allreducing");
        /* Allreduce local tile sizes */
        auto const& tile_inds = this->proc_map->get_my_tile_inds();
        for (int i=0; i<tile_inds.size(); i++)
        {
            auto& p = tile_inds[i];
            tile_sizes[p.first * ntiles + p.second] = local_matrices[i].get_nnz();
        }
        MPI_Allreduce(MPI_IN_PLACE, tile_sizes.data(), tile_sizes.size(), MPIType<IT>(), MPI_SUM, 
                        this->proc_map->get_world_comm());

        DEBUG_PRINT("Done");

    }


    int row_edge_size(const int tidx)
    {
        if (this->proc_map->get_col_rank() != this->proc_map->get_px() - 1)
            return 0;
        int tiles_along_dim = static_cast<int>(std::sqrt(this->n_local_tiles));
        if (tidx / tiles_along_dim == (tiles_along_dim - 1))
            return m - (( m / mb) * mb) ;
        return 0;
    }


    int col_edge_size(const int tidx)
    {
        if (this->proc_map->get_row_rank() != this->proc_map->get_py() - 1)
            return 0;
        int tiles_along_dim = static_cast<int>(std::sqrt(this->n_local_tiles));
        if (tidx % tiles_along_dim == (tiles_along_dim - 1))
            return n - (( n / nb) * nb) ;
        return 0;
    }


    inline void set_rows(const IT _m) {m=_m;}
    inline void set_cols(const IT _n) {n=_n;}
    inline void set_nnz(const IT _nnz) {nnz=_nnz;}

    inline IT get_rows() {return m;}
    inline IT get_cols() {return n;}
    inline IT get_nnz() {return nnz;}

    inline std::vector<SpMat<IT, DT>> get_local_matrices() {return local_matrices;}
    inline int get_n_local_tiles() {return n_local_tiles;}

    

    std::shared_ptr<P> proc_map;

protected:
    IT m, n, nnz;
    std::vector<IT> tile_sizes;

    IT mb, nb;
    IT mtiles, ntiles;
    std::vector<SpMat<IT, DT>> local_matrices;
    int n_local_tiles;

};


template <typename IT, typename DT, typename P>
class DistSpMatCyclic2D : public DistSpMatCyclic<IT, DT, P>
{
public:

    using Triple = std::tuple<IT, IT, DT>;

    DistSpMatCyclic2D(const IT m, const IT n, const IT nnz,
                    const IT mb, const IT nb, 
                    std::shared_ptr<P> proc_map):
        DistSpMatCyclic<IT, DT, P>(m, n, nnz, mb, nb, proc_map)
    {
        this->n_local_tiles = (this->mtiles / this->proc_map->get_px()) * (this->ntiles / this->proc_map->get_py());
    }


    Triple map_glob_to_local(const Triple& t)
    {
        return t;
    }


    /* This maps a triple with global coordinates to local tile coordinates */
    Triple map_glob_to_tile(const Triple& t) override
    {

        IT loc_i = std::get<0>(t) % this->mb;
        IT loc_j = std::get<1>(t) % this->nb;

        IT mp = (this->m / this->mb) * this->mb;
        IT np = (this->n / this->nb) * this->nb;

        if (std::get<0>(t) >= mp)
            loc_i += (std::get<0>(t) - mp);
        if (std::get<1>(t) >= np)
            loc_j += (std::get<1>(t) - np);
        
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
    int tile_owner(const Triple& t) override
    {
        IT i = std::get<0>(t);
        IT j = std::get<1>(t);

        int row_contrib = std::min(i / (this->mb * this->proc_map->get_px()), 
                                  (this->mtiles / this->proc_map->get_px()) - 1) 
                            * (this->ntiles / this->proc_map->get_py());
        int col_contrib = std::min(j / (this->nb * this->proc_map->get_py()), 
                                    (this->ntiles / this->proc_map->get_py()) - 1);

        assert((row_contrib + col_contrib) < this->n_local_tiles);

        return row_contrib + col_contrib;
    }


};

}






#endif
