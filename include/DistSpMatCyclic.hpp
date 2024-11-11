#ifndef DIST_SP_MAT_CYCLIC_HPP
#define DIST_SP_MAT_CYCLIC_HPP

#include "common.h"
#include "ProcMap.hpp"
#include "CooTriples.hpp"
#include "dCSR.cuh"
#include "dCSR_utils.cuh"
#include "SpMat.hpp"
#include "TileWindow.hpp"

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
        mtiles(std::ceil((double)m / (double)mb)), 
        ntiles(std::ceil((double)n / (double)nb)), 
        proc_map(proc_map)
    {
        assert(mb <= (m / proc_map->get_px()));
        assert(nb <= (n / proc_map->get_py()));
        tile_nnz.resize(mtiles * ntiles);
        tile_rows.resize(mtiles * ntiles);
        tile_cols.resize(mtiles * ntiles);
        window_offsets.resize(mtiles * ntiles);
    }


    virtual int tile_owner(const Triple& t) {assert(false);}
    virtual Triple map_glob_to_tile(const Triple& t) {assert(false);}


    void set_from_coo(CooTriples<IT, DT> * triples, const bool triangular=false, 
                        const bool transpose=false)
    {

        if (transpose)
            std::swap(this->mb, this->nb);

        /* Map each triple to the tile that owns it */
        std::vector<CooTriples<IT, DT>> tile_triples(this->n_local_tiles);

        DEBUG_PRINT("Mapping to tiles");

        std::for_each(triples->begin(), triples->end(),
            [&](auto& t)
            {
                int target_tile = this->tile_owner(t);
                if (triangular && proc_map->is_upper_triangular(target_tile))
                    return;
                tile_triples[target_tile].add_triple(this->map_glob_to_tile(t));
            }
        );

        set_from_coo(tile_triples, transpose);

    }


    void set_from_coo(std::vector<CooTriples<IT, DT>>& tile_triples, const bool transpose)
    {
        DEBUG_PRINT("Making local size arrays");

        auto const& tile_inds = this->proc_map->get_my_tile_inds();

        IT row_size = mb;
        IT col_size = nb;

        uint64_t window_size = 0;
        /* Build the CSR arrays for each tile */
        for (int i=0; i<tile_triples.size(); i++) 
        {
            auto& p = tile_inds[i];

            tile_nnz[p.first * ntiles + p.second] = tile_triples[i].get_nnz();
#ifdef DEBUG
            logptr->OFS()<<"nnz in this tile: "<<tile_triples[i].get_nnz()<<std::endl;
#endif
            tile_rows[p.first * ntiles + p.second] = row_size;
            tile_cols[p.first * ntiles + p.second] = col_size;

            window_size += aligned_tile_size(tile_triples[i].get_nnz(), 
                                                row_size);
        }



        DEBUG_PRINT("Allreducing");
        MPI_Allreduce(MPI_IN_PLACE, 
                      tile_nnz.data(), tile_nnz.size(), 
                      MPIType<IT>(), MPI_SUM, 
                      this->proc_map->get_world_comm());
        MPI_Allreduce(MPI_IN_PLACE, 
                      tile_rows.data(), tile_rows.size(), 
                      MPIType<IT>(), MPI_SUM, 
                      this->proc_map->get_world_comm());
        MPI_Allreduce(MPI_IN_PLACE, 
                      tile_cols.data(), tile_cols.size(), 
                      MPIType<IT>(), MPI_SUM, 
                      this->proc_map->get_world_comm());

#ifdef DEBUG
        //logptr->log_vec(tile_nnz, "tile nnz");
        //logptr->log_vec(tile_rows, "tile rows");
        //logptr->log_vec(tile_cols, "tile cols");
        //logptr->log_vec(tile_inds, "tile inds");
#endif


        DEBUG_PRINT("Making tile window");
        this->tile_window = std::make_shared<TileWindow<IT, DT>>(window_size);

        DEBUG_PRINT("Placing tiles");
        place_tiles(tile_triples, transpose);

        DEBUG_PRINT("Done");

    }


    void place_tiles(std::vector<CooTriples<IT, DT>>& tile_triples, const bool transpose)
    {
        auto const& tile_inds = this->proc_map->get_my_tile_inds();
        for (int i=0; i<tile_triples.size(); i++)
        {
            auto& p = tile_inds[i];
            auto offset = tile_window->add_tile(tile_triples[i], 
                                            mb, 
                                            nb, 
                                            transpose);
            window_offsets[p.first * ntiles + p.second] = offset;
        }
        
#ifdef DEBUG
        logptr->log_vec(window_offsets, "Window offsets preallreduce");
        logptr->newline();
#endif

        MPI_Allreduce(MPI_IN_PLACE, 
                      window_offsets.data(), window_offsets.size(), 
                      MPIType<uint64_t>(), MPI_SUM, 
                      this->proc_map->get_world_comm());
#ifdef DEBUG
        logptr->log_vec(window_offsets, "Window offsets");
        logptr->newline();
#endif
    }

    
    SpMat<IT, DT> get_tile_sync(int i, int j)
    {

        //TODO: If local tile, just set pointers without landing zone

        /* Which process owns tile (i,j)? */
        int target_pe = proc_map->get_tile_owners()[i][j];

        /* Where does that tile live in target's TileWindow? */
        uint64_t offset = window_offsets[i * ntiles + j];

#ifdef DEBUG
        logptr->OFS()<<"Target PE: "<<target_pe<<std::endl;
        logptr->OFS()<<"Offset: "<<offset<<std::endl;
#endif


        /* How large is that tile? */
        uint64_t landing_zone_size = aligned_tile_size(tile_nnz[i*ntiles + j], tile_rows[i*ntiles + j]);

        /* Remote tile is empty */
        if (tile_nnz[i*ntiles + j]==0)
        {
            return SpMat<IT, DT>();
        }

        /* Allocate landing zone for the tile */
        char * d_landing_zone;
        CUDA_CHECK(cudaMalloc(&d_landing_zone, landing_zone_size));

        /* Register landing zone in the symmetric heap */
        NVSHMEM_CHECK(nvshmemx_buffer_register(d_landing_zone, landing_zone_size));

        /* Use NVSHMEM to fetch the tile */
        tile_window->get_tile_sync(d_landing_zone, offset, landing_zone_size, target_pe);

        /* Unregister landing zone */
        NVSHMEM_CHECK(nvshmemx_buffer_unregister(d_landing_zone));

        /* Make the SpMat from the landing zone */
        return SpMat<IT, DT>(tile_rows[i*ntiles + j], tile_cols[i*ntiles + j], tile_nnz[i*ntiles + j],
                                d_landing_zone);

    }


    uint64_t aligned_tile_size(const IT nnz, const IT m)
    {
        if (nnz==0) return 0;
        auto vals_size = aligned_offset<DT>(nnz * sizeof(DT));
        auto colinds_size = aligned_offset<DT>(nnz * sizeof(IT));
        auto rowptrs_size = aligned_offset<DT>((m+1)*sizeof(IT)); //align for next tile
        return vals_size + colinds_size + rowptrs_size;
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

    inline IT get_mb() {return mb;}
    inline IT get_nb() {return nb;}

    inline IT get_mtiles() {return mtiles;}
    inline IT get_ntiles() {return ntiles;}

    inline std::vector<SpMat<IT, DT>> get_local_matrices() {return tile_window->get_local_matrices();}
    inline std::shared_ptr<TileWindow<IT, DT>> get_tile_window() {return tile_window;}
    inline int get_n_local_tiles() {return n_local_tiles;}
    

    std::shared_ptr<P> proc_map;

protected:
    IT m, n, nnz;
    std::vector<IT> tile_nnz;
    std::vector<IT> tile_rows;
    std::vector<IT> tile_cols;


    IT mb, nb;
    IT mtiles, ntiles;
    int n_local_tiles;

    std::shared_ptr<TileWindow<IT, DT>> tile_window;
    std::vector<uint64_t> window_offsets;

};


template <typename IT, typename DT, typename P>
bool operator==(DistSpMatCyclic<IT, DT, P>& lhs, DistSpMatCyclic<IT, DT, P>& rhs) 
{
    auto const& lhs_tiles = lhs.get_tile_window()->get_local_matrices();
    auto const& rhs_tiles = rhs.get_tile_window()->get_local_matrices();

    assert (lhs_tiles.size() == rhs_tiles.size());
    int t = lhs_tiles.size();

    bool correct = true;

    MPI_Barrier(lhs.proc_map->get_world_comm());

    for (int i=0; i<t; i++)
    {
        nvshmem_barrier_all();

        auto const& lhs_tile = lhs_tiles[i];
        auto const& rhs_tile = rhs_tiles[i];

        //assert(lhs_tile.get_nnz() == rhs_tile.get_nnz());

        if (lhs_tile.get_nnz()==0 || rhs_tile.get_nnz()==0) continue;

        bool equals = (lhs_tile == rhs_tile);

        correct = correct && equals;

    }

    int correct_int = correct ? 1 : 0;

    MPI_Allreduce(MPI_IN_PLACE, &correct_int, 1, MPI_INT, MPI_LAND, lhs.proc_map->get_world_comm());

    return true ? (correct_int) : false;
}


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
        MPI_Barrier(proc_map->get_world_comm());
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

        /*
        IT mp = (this->m / this->mb) * this->mb;
        IT np = (this->n / this->nb) * this->nb;

        if (std::get<0>(t) >= mp)
            loc_i += (std::get<0>(t) - mp);
        if (std::get<1>(t) >= np)
            loc_j += (std::get<1>(t) - np);
            */
#ifdef DEBUG
        logptr->OFS()<<"i: "<<std::get<0>(t)<<", j: "<<std::get<1>(t)<<std::endl;
        logptr->OFS()<<"i: "<<loc_i<<", j: "<<loc_j<<std::endl;
#endif
        
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
