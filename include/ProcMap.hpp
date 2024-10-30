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

        /* Coll comm */
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

private:

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

//TODO: Block cyclic map



}


#endif
