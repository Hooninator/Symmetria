#ifndef SPSYRK_CUH
#define SPSYRK_CUH

#include "../common.h"
#include "../DistSpMat.hpp"

#include "galatic_spgemm.cuh"


namespace symmetria {
template <typename SR, typename IT, typename DT>
DistSpMat1DBlockRow<IT, DT> spsyrk_bulksync_1d_rowblock(DistSpMat1DBlockRow<IT, DT>& A)
{

    /* Bookkeeping */
    auto proc_map = A.proc_map;
    const int p = proc_map->get_n_procs();

    /* These vectors will be resized each iteration so they can hold
     * local tiles.
     */
    std::vector<DT> vals;
    std::vector<IT> colinds;
    std::vector<IT> rowptrs;
    

    /* Need array of communicators for each subset of the processes that are broadcast to at each stage */
    std::vector<MPI_Comm> comms(p);

    std::vector<int> procs_in_group(p);
    std::iota(procs_in_group.begin(), procs_in_group.end(), 0);

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    for (int i=0; i<p; i++)
    {
        MPI_Group group;
        MPI_Group_incl(world_group, (p - i), procs_in_group.data() + i, &group);
        MPI_Comm_create(MPI_COMM_WORLD, world_group, &comms[i]);
    }

    /* Create transposed version of my local block row */
        

    /* Main loop */
    for (int k=0; k<p; k++)
    {

        /* Non-blocking broadcast of tranposed block row k */

        /* If first iteration, overlap broadcasts with computing diagonal blocks */

        /* Wait on bcast completion */

        /* If rank > k, multiply the tile I just received */

        /* Push output tuples to host vector */

    }


    /* Merge output tuples */

    /* Return final matrix */

}

}


#endif
