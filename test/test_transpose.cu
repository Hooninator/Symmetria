#include "Symmetria.hpp"
#include "dCSR.cuh"
#include "SemiRingInterface.h"
#include "../source/device/Multiply.cuh"

#include "test_utils.cuh"

using namespace symmetria;

int main(int argc, char ** argv)
{
    typedef int64_t IT;
    typedef double DT;

    symmetria_init();

    {
        const uint32_t m = 16;
        const uint32_t n = 16;
        const uint32_t nnz = 16;

        TEST_LOG("Initialized symmetria");

        std::shared_ptr<ProcMap> proc_map = std::make_shared<ProcMap>(n_pes, MPI_COMM_WORLD);
        TEST_LOG("Initialized process map");

        DistSpMat1DBlockRow<IT, DT> A(m, n, nnz, proc_map);
        symmetria::io::read_mm<IT, DT>("../test/test_matrices/n16.mtx", A);

        MPI_Barrier(MPI_COMM_WORLD);

        /* Correct transposed matrix */
        DistSpMat1DBlockRow<IT, DT> A_t_correct(m, n, nnz, proc_map);
        symmetria::io::read_mm<IT, DT>("../test/test_matrices/n16_transpose.mtx", A_t_correct);

        TEST_LOG("Read in matrices");
        MPI_Barrier(MPI_COMM_WORLD);

        /* Make dCSR from local data */
        dCSR<DT> A_dcsr;
        make_dCSR_from_distspmat(A, A_dcsr);

        dCSR<DT> A_dscr_correct;
        make_dCSR_from_distspmat(A_t_correct, A_dscr_correct);

        /* Transpose */
        auto A_t = transpose_outofplace(A_dcsr);

        /* Correctness */
        bool is_correct = (A_t == A_dscr_correct);

        TEST_CHECK(is_correct);
    }

    symmetria_finalize();
	
	TEST_SUCCESS("Transpose");
    return 0;
}
