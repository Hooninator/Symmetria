#include "Symmetria.hpp"

using namespace symmetria;

int main(int argc, char ** argv)
{
    typedef int64_t IT;
    typedef double DT;

    symmetria_init();

    const uint32_t m = 213360;
    const uint32_t n = 213360;
    const uint32_t nnz = 3021648;

    std::shared_ptr<ProcMap> proc_map = std::make_shared<ProcMap>(4, MPI_COMM_WORLD);
    DistSpMat1DBlockRow<IT, DT> A(m, n, nnz, proc_map);
    symmetria::io::read_mm<IT, DT>("/pscratch/sd/j/jbellav/matrices/stomach/stomach.mtx", A);

    spsyrk_bulksync_1d_rowblock<PlusTimesSemiring>(A);

    symmetria_finalize();
    return 0;
}
