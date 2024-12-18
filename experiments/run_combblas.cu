#include "Symmetria.hpp"
#include "experiment_common.cuh"

#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"

using namespace combblas;

#define THREADED
#define _OPENMP

#ifdef TIMING
double cblas_alltoalltime;
double cblas_allgathertime;
#endif

int cblas_splits = omp_get_max_threads();

#define ElementType double

// Simple helper class for declarations: Just the numerical type is templated
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat
{
public:
	typedef SpDCCols<uint32_t, NT> DCCols;
	typedef SpParMat<uint32_t, NT, DCCols> MPI_DCCols;
};


int main(int argc, char ** argv)
{
    std::vector<const char *> req_args {"--rows", "--cols", "--nnz", "--ntrials", "--name", "--type"};
    ExperimentConfig config;
    config.parse_args(argc, argv, req_args);

	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	{
        std::string path("../experiments/"+config.name+"/"+config.name+".mtx");

		string Aname(path);
		string Bname(path);

		MPI_Barrier(MPI_COMM_WORLD);
		typedef PlusTimesSRing<double, double> PTDOUBLEDOUBLE;
		typedef SelectMaxSRing<bool, int64_t> SR;

		shared_ptr<CommGrid> fullWorld;
		fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));

		// construct objects
		PSpMat<double>::MPI_DCCols A(fullWorld);
		PSpMat<double>::MPI_DCCols B(fullWorld);
		PSpMat<double>::MPI_DCCols C(fullWorld);

		A.ParallelReadMM(Aname, true, maximum<double>());
        B = A;
        B.Transpose();

		A.PrintInfo();

        // Timers
        symmetria::Timer timer;

        timer.start_timer("CPUMult");
		C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols>(A, B);
		MPI_Barrier(MPI_COMM_WORLD);
        timer.stop_timer("CPUMult");

		std::cout << "Time for CPU mult: " << timer.get_timer("CPUMult") << std::endl;
		C.PrintInfo();
		MPI_Barrier(MPI_COMM_WORLD);

        /*
        timer.start_timer("GPUMult");
		C = Mult_AnXBn_DoubleBuff_CUDA<PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols>(A, B);
		MPI_Barrier(MPI_COMM_WORLD);
        timer.stop_timer("GPUMult");

		std::cout << "Time for GPU mult: " << timer.get_timer("GPUMult") << std::endl;
		C.PrintInfo();
		MPI_Barrier(MPI_COMM_WORLD);
        */

        const std::string csv_name("timings_combblas_2d_"+config.name+"_"+STR(myrank)+".csv");
        timer.write_all_timers(csv_name, std::ofstream::trunc);


	}
	MPI_Finalize();

    return 0;
}
