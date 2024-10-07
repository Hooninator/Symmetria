#ifndef MPI_UTILS_CUH
#define MPI_UTILS_CUH
#include "../common.h"

namespace symmetria {

namespace mpi {

/* Non blocking broadcast of tile of matrix that lives on root.
 */
template <typename IT, typename IT2, typename DT>
void ibcast_tile(const int root, 
                 const MPI_Comm& comm,
                 DT * vals,
                 IT * colinds,
                 IT * rowptrs,
                 MPI_Request * requests,
                 const IT2 nnz, const IT2 rows)
{

    MPI_Ibcast(vals, nnz, MPIType<DT>(),
               root, comm,
               &requests[0]);

    MPI_Ibcast(colinds, nnz, MPIType<IT>(),
               root, comm,
               &requests[1]);

    MPI_Ibcast(rowptrs, rows + 1, MPIType<IT>(),
               root, comm,
               &requests[2]);

}

}
}


#endif
