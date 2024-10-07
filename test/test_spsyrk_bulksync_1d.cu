#include "Symmetria.hpp"
#include "dCSR.cuh"
#include "CSR.cuh"
#include "CPU_SpGEMM.h"
#include "SemiRingInterface.h"
#include "../source/device/Multiply.cuh"

#include "TestDriver.hpp"

//#define DEBUG_TEST
#define EPS 0.001

using namespace symmetria;
using namespace symmetria::testing;


class TestSpSYRKSync1D : public TestDriver<TestSpSYRKSync1D>
{
public:

    bool run_test_impl(TestParams& test)
    {
        typedef unsigned int IT;
        typedef double DT;

        const uint32_t m = test.rows;
        const uint32_t n = test.cols;
        const uint32_t nnz = test.nnz;
        const std::string name(test.name);
        const std::string path(std::string("../test/test_matrices/")+test.name);

        std::shared_ptr<ProcMap> proc_map = std::make_shared<ProcMap>(n_pes, MPI_COMM_WORLD);

        /* Read in matrix */
        DistSpMat1DBlockRow<IT, DT> A(m, n, nnz, proc_map);
        symmetria::io::read_mm<IT, DT>(path.c_str(), A);
        MPI_Barrier(MPI_COMM_WORLD);

        /* Do SpSYRK */
        using Semiring = PlusTimesSemiring<DT>;
        auto C_computed = spsyrk_bulksync_1d_rowblock<Semiring>(A);

        TEST_PRINT("Done with SpSYRK");

        /* TODO: Correctness check */

        return true;
    };
};




int main()
{
    symmetria_init();
    {
        TestDriver<TestSpSYRKSync1D> manager("../test/test_configs.json", "SpSYRK Sync 1D");
        manager.run_tests();
    }
    symmetria_finalize();
	
    return 0;
}
