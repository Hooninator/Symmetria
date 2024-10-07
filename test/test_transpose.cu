#include "Symmetria.hpp"
#include "dCSR.cuh"
#include "SemiRingInterface.h"
#include "../source/device/Multiply.cuh"

#include "TestDriver.hpp"

#define DEBUG_TEST
#define EPS 0.001


using namespace symmetria;
using namespace testing;

Log * test_logptr;

class TestTranspose : public TestDriver<TestTranspose>
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
        const std::string path(std::string("../test/test_matrices/")+test.name+".mtx");

        std::shared_ptr<ProcMap> proc_map = std::make_shared<ProcMap>(n_pes, MPI_COMM_WORLD);

        /* Read in matrix */
        DistSpMat1DBlockRow<IT, DT> A(m, n, nnz, proc_map);
        symmetria::io::read_mm<IT, DT>(path.c_str(), A);
        MPI_Barrier(MPI_COMM_WORLD);

        /* Local submatrices */
        dCSR<DT> A_dcsr = make_dCSR_from_distspmat_outofplace<DT>(A);
        dCSR<DT> A_t_correct = cpu_transpose(A_dcsr);

        /* Do transpose */
        auto A_t = transpose_outofplace(A_dcsr);

#ifdef DEBUG_TEST
        test_logptr->OFS()<<"A"<<std::endl;
        dump_dCSR_to_log(test_logptr, A_dcsr);
        test_logptr->OFS()<<"At_correct"<<std::endl;
        dump_dCSR_to_log(test_logptr, A_t_correct);
        test_logptr->OFS()<<"At_actual"<<std::endl;
        dump_dCSR_to_log(test_logptr, A_t);
#endif

        CSR<DT> h_A_t;
        CSR<DT> h_A_t_correct;

        convert(h_A_t, A_t);
        convert(h_A_t_correct, A_t_correct);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto triples_correct = CooTriples(h_A_t_correct.data.get(), 
                                          h_A_t_correct.col_ids.get(),
                                          h_A_t_correct.row_offsets.get(),
                                          (IT)h_A_t_correct.nnz, (IT)h_A_t_correct.rows);

        auto triples_actual = CooTriples(h_A_t.data.get(), 
                                          h_A_t.col_ids.get(),
                                          h_A_t.row_offsets.get(),
                                          (IT)h_A_t.nnz, (IT)h_A_t.rows);

        triples_correct.rowsort();
        triples_correct.rowsort();
        triples_actual.rowsort();
        triples_actual.rowsort();

        auto triple_comp = [](auto& t1, auto& t2) 
        {
            return (std::get<0>(t1)==std::get<0>(t2) &&
                    std::get<1>(t1)==std::get<1>(t2) &&
                    fabs(std::get<2>(t1) - std::get<2>(t2)) < EPS);
        };

        TEST_CHECK(compare_vectors(triples_correct, triples_actual, triple_comp));

        dealloc(A_t);
        dealloc(A_t_correct);
        
        return true;
    }

    template <typename T>
    dCSR<T> cpu_transpose(dCSR<T>& A)
    {
        std::vector<T> h_vals(A.nnz);
        std::vector<unsigned int> h_colinds(A.nnz);
        std::vector<unsigned int> h_rowptrs(A.rows + 1);

        CUDA_CHECK(cudaMemcpy(h_vals.data(), A.data, sizeof(T)*h_vals.size(),
                                cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_colinds.data(), A.col_ids, sizeof(unsigned int )*A.nnz,
                                cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rowptrs.data(), A.row_offsets, sizeof(unsigned int) * (A.rows + 1),
                                cudaMemcpyDeviceToHost));

        unsigned int num_cols = A.cols;
        unsigned int num_rows = A.rows;

        std::vector<T> h_trans_vals(A.nnz);
        std::vector<unsigned int> h_trans_colinds(A.nnz);
        std::vector<unsigned int> h_trans_rowptrs(A.cols + 1);

        // Initialize row pointers for the transposed matrix
        h_trans_rowptrs.resize(num_cols + 1, 0);

        // Step 1: Compute the number of elements in each column (which becomes row in transposed matrix)
        for (int i = 0; i < h_colinds.size(); ++i) {
            h_trans_rowptrs[h_colinds[i] + 1]++;
        }

        // Step 2: Compute the cumulative row pointer for the transposed matrix
        for (int i = 0; i < num_cols; ++i) {
            h_trans_rowptrs[i + 1] += h_trans_rowptrs[i];
        }

        // Step 3: Initialize the values and columns for the transposed matrix
        h_trans_vals.resize(h_vals.size());
        h_trans_colinds.resize(h_colinds.size());

        // Step 4: Fill the values and columns for the transposed matrix
        std::vector<int> col_counts(num_cols, 0);  // Temporary array to track element position in rows

        for (int row = 0; row < num_rows; ++row) {
            for (int j = h_rowptrs[row]; j < h_rowptrs[row + 1]; ++j) {
                int col = h_colinds[j];
                int dest_pos = h_trans_rowptrs[col] + col_counts[col];

                h_trans_vals[dest_pos] = h_vals[j];
                h_trans_colinds[dest_pos] = row;
                
                col_counts[col]++;
            }
        }

        dCSR<T> A_t;
        A_t.alloc(num_cols, num_rows, A.nnz);

        CUDA_CHECK(cudaMemcpy(A_t.data, h_trans_vals.data(), sizeof(T)*A.nnz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(A_t.col_ids, h_trans_colinds.data(), sizeof(unsigned int)*A.nnz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(A_t.row_offsets, h_trans_rowptrs.data(), sizeof(unsigned int)*(A.cols + 1), cudaMemcpyHostToDevice));

        return A_t;
    }
};



        



int main(int argc, char ** argv)
{

    symmetria_init();

#ifdef DEBUG_TEST
    std::stringstream ss;
    ss<<"TestLogfile"<<symmetria::my_pe<<".out";
    test_logptr = new Log(ss.str().c_str());
#endif

    {
        TestDriver<TestTranspose> manager("../test/test_configs.json", "Transpose");
        manager.run_tests();
    }

#ifdef DEBUG_TEST
    delete test_logptr;
#endif

    symmetria_finalize();
	
    return 0;
}
