#include "Symmetria.hpp"
#include "experiment_common.cuh"

using namespace symmetria;

typedef unsigned int IT;
typedef double DT;

void run_spsyrk_1d(ExperimentConfig& config)
{

    symmetria_init();
    {

        /* Config */
        std::string path("../experiments/suitesparse/"+config.name+"/"+config.name+".mtx");
        unsigned int m = config.rows;
        unsigned int n = config.cols;
        unsigned int nnz = config.nnz;
        unsigned int ntrials = config.ntrials;
        std::shared_ptr<ProcMap> proc_map = std::make_shared<ProcMap>(symmetria::n_pes, MPI_COMM_WORLD);

        /* Read in matrix */
        DistSpMat1DBlockRow<IT, DT> A(m, n, nnz, proc_map);
        symmetria::io::read_mm<IT, DT>(path.c_str(), A);
        MPI_Barrier(MPI_COMM_WORLD);

        /* Do SpSYRK */
        using Semiring = PlusTimesSemiring<DT>;
        
        timer_ptr->start_timer("SpSYRK");
        auto C_computed = spsyrk_bulksync_1d_rowblock<Semiring>(A);
        timer_ptr->stop_timer("SpSYRK");

        /* Write timer to JSON */
        std::string json_name("timings_spsyrk_1d_"+STR(symmetria::my_pe)+".json");
        timer_ptr->write_all_timers(json_name);

    }
    symmetria_finalize();

}

int main(int argc, char ** argv)
{

    ExperimentConfig config = parse_args(argc, argv);
    std::string type(config.type);

    if (type.compare("1d")==0) {
        std::cout<<"Running 1D experiment"<<std::endl;
        run_spsyrk_1d(config);
    }

    return 0;
}
