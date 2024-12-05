#include "Symmetria.hpp"
#include "dCSR.cuh"
#include "CSR.cuh"
#include "CPU_SpGEMM.h"
#include "SemiRingInterface.h"
#include "../source/device/Multiply.cuh"

#include "TestDriver.hpp"

//#define DEBUG
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
        const uint32_t mb = std::floor((float)m / (2*(int)sqrt(n_pes)));
        const uint32_t nb = std::floor((float)n / (2*(int)sqrt(n_pes)));
        const uint32_t mtiles = std::floor( (float)m / (float)mb );
        const uint32_t ntiles = std::floor( (float)n / (float)nb );

        TEST_PRINT("M: " + std::to_string(m) + " N: " + std::to_string(n) + " NNZ: " + std::to_string(nnz) + " MB: " + std::to_string(mb) + " NB: " + std::to_string(nb));

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

        using Semiring = PlusTimesSemiring<DT>;

        auto C = spsyrk_cyclic_2d<Semiring>(A);

        TEST_PRINT("Done with SpSYRK");

        /* Correctness check */
#ifdef DEBUG
        logptr->OFS()<<"START READING CORRECT"<<std::endl;
#endif
        DistSpMatCyclic2D<IT, DT, ProcMapCyclic2D> C_correct(m, m, C.get_nnz(), mb, mb, proc_map);
        symmetria::io::read_mm<IT, DT>(product_path.c_str(), C_correct, true);
        TEST_PRINT("Done reading in correct");

        TEST_CHECK(C_correct == C);
        TEST_PRINT("Done");

        return true;
    };
};




int main(int argc, char ** argv)
{

    std::string test_name = "none";
    if (argc > 1)
        test_name = std::string((argv[1]));

    symmetria_init();
    {
        TestDriver<TestSpSYRKCyclic2D> manager("../test/test_configs.json", "SpSYRK Cyclic 2D");
        manager.run_tests(test_name.c_str());
    }
    symmetria_finalize();
	
    return 0;
}
