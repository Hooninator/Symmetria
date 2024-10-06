#include "Symmetria.hpp"
#include "dCSR.cuh"
#include "CSR.cuh"
#include "CPU_SpGEMM.h"
#include "SemiRingInterface.h"
#include "../source/device/Multiply.cuh"

#include "test_utils.cuh"

//#define DEBUG_TEST
#define EPS 0.001

using namespace symmetria;

template <typename IT, typename DT>
bool compare(CooTriples<IT, DT>& h_correct_triples, std::tuple<IT, IT, DT> * d_triples, const IT nnz)
{

    std::vector<std::tuple<IT, IT, DT>> h_actual_triples(nnz);

    CUDA_CHECK(cudaMemcpy(h_actual_triples.data(), d_triples, sizeof(std::tuple<IT, IT, DT>)* nnz,
                            cudaMemcpyDeviceToHost));

    auto triple_comp = [](auto& t1, auto& t2) 
    {
        return (std::get<0>(t1)==std::get<0>(t2) &&
                std::get<1>(t1)==std::get<1>(t2) &&
                fabs(std::get<2>(t1) - std::get<2>(t2)) < EPS);
    };

    return test_utils::compare_vectors(h_correct_triples, h_actual_triples, triple_comp);
}


void run_test();


int main(int argc, char ** argv)
{
    typedef unsigned int IT;
    typedef double DT;

    symmetria_init();

    {

        const uint32_t m = 16;
        const uint32_t n = 16;
        const uint32_t nnz = 16;

        std::shared_ptr<ProcMap> proc_map = std::make_shared<ProcMap>(n_pes, MPI_COMM_WORLD);

        /* Read in matrix */
        DistSpMat1DBlockRow<IT, DT> A(m, n, nnz, proc_map);
        symmetria::io::read_mm<IT, DT>("../test/test_matrices/n16.mtx", A);
        MPI_Barrier(MPI_COMM_WORLD);

        /* Fetch local submatrices */
        dCSR<DT> A_dcsr = make_dCSR_from_distspmat_outofplace<DT>(A);
        dCSR<DT> A_t = transpose_outofplace(A_dcsr);

        /* GPU multiply */
        using Semiring = PlusTimesSemiring<DT>;
        IT nnz_C;
        auto d_C = local_spgemm_galatic<Semiring>(A_dcsr, A_t, nnz_C);

        /* Do CPU multiply for correctness check */
        CSR<DT> h_A;
        CSR<DT> h_A_t;
        CSR<DT> h_C_csr;

        convert(h_A, A_dcsr);
        convert(h_A_t, A_t);
        CUDA_CHECK(cudaDeviceSynchronize());

        Semiring semiring;
        Mult_CPU<Semiring>(h_A, h_A_t, h_C_csr, semiring);

        CooTriples<IT, DT> h_C_triples(h_C_csr.data.get(), h_C_csr.col_ids.get(), 
                                        h_C_csr.row_offsets.get(), 
                                        h_C_csr.nnz, h_C_csr.rows);
#ifdef DEBUG_TEST
        h_C_triples.dump_to_log(logptr, "Correct output");
#endif

        TEST_CHECK(((size_t)nnz_C == h_C_csr.nnz));
        TEST_CHECK(compare(h_C_triples, d_C, nnz_C));

        CUDA_CHECK(cudaFree(d_C));
    }

    TEST_SUCCESS("Local Mult");

    symmetria_finalize();
	
    return 0;
}