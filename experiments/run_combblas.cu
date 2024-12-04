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

#ifdef TIMING
double cblas_alltoalltime;
double cblas_allgathertime;
#endif

#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif

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
    std::vector<const char *> req_args {"--ntrials", "--name"};

    ExperimentConfig config = parse_args(argc, argv, req_args);

	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	{
        std::string path("../experiments/suitesparse/"+config.name+"/"+config.name+".mtx");

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
		B.ParallelReadMM(Bname, true, maximum<double>());

		A.PrintInfo();

		double t1 = MPI_Wtime();
		C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols>(A, B);
		MPI_Barrier(MPI_COMM_WORLD);
		double t2 = MPI_Wtime();

		std::cout << "Time for CPU mult: " << t2 - t1 << std::endl;
		C.PrintInfo();
		MPI_Barrier(MPI_COMM_WORLD);

		t1 = MPI_Wtime();
		C = Mult_AnXBn_DoubleBuff_CUDA<PTDOUBLEDOUBLE, double, PSpMat<double>::DCCols>(A, B);
		MPI_Barrier(MPI_COMM_WORLD);
		t2 = MPI_Wtime();

		std::cout << "Time for GPU mult: " << t2 - t1 << std::endl;
		C.PrintInfo();
		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Pcontrol(1, "SpGEMM_DoubleBuff");
		t1 = MPI_Wtime(); // initilize (wall-clock) timer
		for (int i = 0; i < config.ntrials; i++)
		{
			C = Mult_AnXBn_DoubleBuff_CUDA<PTDOUBLEDOUBLE, ElementType, PSpMat<ElementType>::DCCols>(A, B);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		t2 = MPI_Wtime();
		MPI_Pcontrol(-1, "SpGEMM_DoubleBuff");
		if (myrank == 0)
		{
			cout << "Double buffered CUDA multiplications finished" << endl;
			printf("%.6lf seconds elapsed per iteration\n", (t2 - t1) / (double)config.ntrials);
		}

		MPI_Pcontrol(1, "SpGEMM_DoubleBuff");
		t1 = MPI_Wtime(); // initilize (wall-clock) timer
		for (int i = 0; i < config.ntrials; i++)
		{
			C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, ElementType, PSpMat<ElementType>::DCCols>(A, B);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		t2 = MPI_Wtime();
		MPI_Pcontrol(-1, "SpGEMM_DoubleBuff");
		if (myrank == 0)
		{
			cout << "Double buffered CPU multiplications finished" << endl;
			printf("%.6lf seconds elapsed per iteration\n", (t2 - t1) / (double)config.ntrials);
		}
	}
	MPI_Finalize();

    return 0;
}
