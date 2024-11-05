#ifndef PROC_MAP_HPP
#define PROC_MAP_HPP

#include "common.h"

namespace symmetria {
    
class ProcMap 
{

public:

    /* Process map used to specify the distribution of a matrix across GPUs. */

    // 1D constructor
    ProcMap(const int x, const MPI_Comm comm):
        px(x), py(1), pz(1), world_comm(comm), grid_size(x)
    {

        MPI_Comm_size(comm, &n_procs);
        assert(n_procs==px*py*pz);

        set_ranks();

        make_comms();

    }


    // 2D constructor
    ProcMap(const int x, const int y, const MPI_Comm comm):
        px(x), py(y), pz(1), world_comm(comm), grid_size(x*y)
    {
        assert(y==x);
        MPI_Comm_size(comm, &n_procs);
        assert(n_procs==px*py*pz);
        set_ranks();
        make_comms();
    }


    // 3D constructor
    ProcMap(const int x, const int y, const int z,
             const MPI_Comm comm):
        px(x), py(y), pz(z), world_comm(comm), grid_size(x*y*z)
    {

        MPI_Comm_size(comm, &n_procs);
        assert(n_procs==px*py*pz);

        set_ranks();

        make_comms();

    }


    void set_ranks()
    {
        MPI_Comm_rank(world_comm, &rank);

        row_rank = rank % py;
        col_rank = rank / py;

        fiber_rank = rank / grid_size;
        grid_rank = rank % grid_size;
    }


    void make_comms()
    {
        int color, key;

        /* Row comm */
        color = rank / py;
        key = row_rank;
        MPI_Comm_split(MPI_COMM_WORLD, color, key, &row_comm);

        /* Col comm */
        color = rank % py;
        key = col_rank;
        MPI_Comm_split(MPI_COMM_WORLD, color, key, &col_comm);

        /* Fiber comm */
        color = rank % grid_size;
        key = fiber_rank;
        MPI_Comm_split(MPI_COMM_WORLD, color, key, &fiber_comm);

        /* Grid comm */
        color = rank / grid_size;
        key = grid_rank;
        MPI_Comm_split(MPI_COMM_WORLD, color, key, &grid_comm);

    }


    void barrier()
    {
        MPI_Barrier(world_comm);
    }


    inline MPI_Comm get_world_comm() const {return world_comm;}
    inline MPI_Comm get_row_comm() const {return row_comm;}
    inline MPI_Comm get_col_comm() const {return col_comm;}
    inline MPI_Comm get_fiber_comm() const {return fiber_comm;}
    inline MPI_Comm get_grid_comm() const {return grid_comm;}

    
    inline int get_n_procs() const {return n_procs;}
    inline int get_grid_size() const {return grid_size;}
    inline int get_px() const {return px;}
    inline int get_py() const {return py;}
    inline int get_pz() const {return pz;}


    inline int get_rank() const {return rank;}
    inline int get_row_rank() const {return row_rank;}
    inline int get_col_rank() const {return col_rank;}
    inline int get_fiber_rank() const {return fiber_rank;}

protected:

    MPI_Comm world_comm;
    MPI_Comm row_comm;
    MPI_Comm col_comm;
    MPI_Comm fiber_comm;
    MPI_Comm grid_comm;

    int n_procs;
    int grid_size;
    int px;
    int py;
    int pz;

    int rank;
    int row_rank;
    int col_rank;
    int fiber_rank;
    int grid_rank;

};


class ProcMapCyclic2D : public ProcMap
{
public:
    ProcMapCyclic2D(const int x, const int y, 
                    const int mtiles, const int ntiles,
                    const MPI_Comm comm):
        ProcMap(x, y, comm),
        mtiles(mtiles), ntiles(ntiles),
        tile_owners(mtiles)
    {
        for (int i=0; i<mtiles; i++)
            tile_owners[i].resize(ntiles);
        //assert(mtiles==ntiles);
        set_tile_owners();
    }


    void set_tile_owners()
    {
        for (int i=0; i<mtiles; i++)
        {
            for (int j=0; j<ntiles; j++)
            {
                int owner = (j % this->py) + (i % this->px) * this->py;
                tile_owners[i][j] = owner;
                if (owner==this->rank)
                    my_tile_inds.push_back({i, j});
            }
        }
    }

    inline int get_mtiles() {return mtiles;}
    inline int get_ntiles() {return ntiles;}

    inline std::vector<std::vector<int>> get_tile_owners() {return tile_owners;}
    inline std::vector<std::pair<int, int>> get_my_tile_inds() {return my_tile_inds;}

private:
    int mtiles, ntiles;
    std::vector<std::vector<int>> tile_owners;
    std::vector<std::pair<int, int>> my_tile_inds;

};


}


#endif
