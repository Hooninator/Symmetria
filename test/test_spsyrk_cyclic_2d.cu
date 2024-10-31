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
        const uint32_t mtiles = std::ceil( (float)m / (float)mb );
        const uint32_t ntiles = std::ceil( (float)n / (float)nb );

        const std::string name(test.name);
        const std::string path(std::string("../test/test_matrices/")+test.name+".mtx");
        const std::string product_path(std::string("../test/test_matrices/")+test.name+"_product.mtx");

        TEST_PRINT("Making proc map");
        std::shared_ptr<ProcMapCyclic2D> proc_map = std::make_shared<ProcMapCyclic2D>((int)sqrt(n_pes), (int)sqrt(n_pes), 
                                                                                        mtiles, ntiles, MPI_COMM_WORLD);
        TEST_PRINT("Made proc map");

        /* Read in matrix */
        DistSpMatCyclic2D<IT, DT, ProcMapCyclic2D> A(m, n, nnz, mb, nb, proc_map);
        TEST_PRINT("Made matrix");

        symmetria::io::read_mm<IT, DT>(path.c_str(), A);
        MPI_Barrier(MPI_COMM_WORLD);
        TEST_PRINT("Done with IO");
        /*

        using Semiring = PlusTimesSemiring<DT>;

        TEST_PRINT("Done with SpSYRK");
        */


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
