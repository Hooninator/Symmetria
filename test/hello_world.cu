#include "Symmetria.hpp"

using namespace symmetria;

int main(int argc, char ** argv)
{
    typedef int64_t IT;
    typedef double DT;

    symmetria_init();

    DistSpMat1DBlockRow<IT, DT> A(4, MPI_COMM_WORLD);
    symmetria::io::read_mm<IT, DT>("/pscratch/sd/j/jbellav/matrices/stomach/stomach.mtx", A);

    spsyrk_bulksync_1d_rowblock(A);

    symmetria_finalize();
    return 0;
}
