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


class TestSpSYRKCyclic2D : public TestDriver<TestSpSYRKCyclic2D>
{
public:

    bool run_test_impl(TestParams& test)
    {
        typedef unsigned int IT;
        typedef double DT;

        const uint32_t m = test.rows;
        const uint32_t n = test.cols;
        const uint32_t nnz = test.nnz;
        const uint32_t mb = 2;
        const uint32_t nb = 2;
        const std::string name(test.name);
        const std::string path(std::string("../test/test_matrices/")+test.name+".mtx");
        const std::string product_path(std::string("../test/test_matrices/")+test.name+"_product.mtx");

        std::shared_ptr<ProcMap> proc_map = std::make_shared<ProcMap>(n_pes, MPI_COMM_WORLD);

        /* Read in matrix */
        DistSpMatCyclic2D<IT, DT> A(m, n, mb, nb, nnz, proc_map);
        symmetria::io::read_mm<IT, DT>(path.c_str(), A);
        MPI_Barrier(MPI_COMM_WORLD);

        /* Do SpSYRK */
        using Semiring = PlusTimesSemiring<DT>;

        TEST_PRINT("Done with SpSYRK");

        /* Correctness check */

        return true;
    };
};




int main(int argc, char ** argv)
{
    int test_id = -1;
    if (argc > 1)
        test_id = std::atoi(argv[1]);

    symmetria_init();
    {
        TestDriver<TestSpSYRKCyclic2D> manager("../test/test_configs.json", "SpSYRK Cyclic 2D", test_id);
        manager.run_tests();
    }
    symmetria_finalize();
	
    return 0;
}
