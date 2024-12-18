#include "Symmetria.hpp"
#include "experiment_common.cuh"

using namespace symmetria;

typedef unsigned int IT;
typedef double DT;

#define THREADED

void run_spsyrk_1d(ExperimentConfig& config)
{
    symmetria_init();
    {

        /* Config */
        std::string path("../experiments/"+config.name+"/"+config.name+".mtx");
        unsigned int m = config.rows;
        unsigned int n = config.cols;
        unsigned int nnz = config.nnz;
        unsigned int ntrials = config.ntrials;


        std::shared_ptr<ProcMap> proc_map = std::make_shared<ProcMap>(n_pes, MPI_COMM_WORLD);

        /* Read in matrix */
        DistSpMat1DBlockRow<IT, DT> A(m, n, nnz, proc_map);
        symmetria::io::read_mm<IT, DT>(path.c_str(), A);
        MPI_Barrier(MPI_COMM_WORLD);

        using Semiring = PlusTimesSemiring<DT>;
        
        /* Do SpSYRK */
        for (int i=0; i<ntrials; i++) 
        {
            timer_ptr->start_timer("SpSYRK");
            auto C_computed = spsyrk_bulksync_1d_rowblock<Semiring>(A);
            timer_ptr->stop_timer("SpSYRK");

            double t = timer_ptr->get_timer("SpSYRK");
            
            if (symmetria::my_pe==0) {
                std::cout<<"Time for SpSYRK: "<<t<<"s"<<std::endl;
                std::cout<<"NNZ C: "<<C_computed.get_nnz()<<std::endl;
            }

            /* Write timer to CSV */
            std::string csv_name("timings_spsyrk_1d_"+config.name+"_"+STR(symmetria::my_pe)+".csv");

            /* If first trial, clear the file, else append */
            if (i==0)
                timer_ptr->write_all_timers(csv_name, std::ofstream::trunc);
            else
                timer_ptr->write_all_timers(csv_name, std::ofstream::app);

            timer_ptr->clear_all_timers();
            proc_map->barrier();
        }

    }
}

void run_spsyrk_2d(ExperimentConfig& config)
{

    symmetria_init();
    {

        /* Config */
        std::string path("../experiments/"+config.name+"/"+config.name+".mtx");
        unsigned int m = config.rows;
        unsigned int n = config.cols;
        unsigned int nnz = config.nnz;
        unsigned int ntrials = config.ntrials;

        const uint32_t mb = std::floor((float)m / (2*(int)sqrt(n_pes)));
        const uint32_t nb = std::floor((float)n / (2*(int)sqrt(n_pes)));
        const uint32_t mtiles = std::floor( (float)m / (float)mb );
        const uint32_t ntiles = std::floor( (float)n / (float)nb );

        std::shared_ptr<ProcMapCyclic2D> proc_map = std::make_shared<ProcMapCyclic2D>(
                (int)sqrt(symmetria::n_pes), 
                (int)sqrt(symmetria::n_pes), 
                mtiles, ntiles,
                MPI_COMM_WORLD);

        /* Read in matrix */
        DistSpMatCyclic2D<IT, DT, ProcMapCyclic2D> A(m, n, nnz, mb, nb, proc_map);
        symmetria::io::read_mm<IT, DT>(path.c_str(), A);
        MPI_Barrier(MPI_COMM_WORLD);

        using Semiring = PlusTimesSemiring<DT>;
        
        /* Do SpSYRK */
        for (int i=0; i<ntrials; i++) 
        {
            timer_ptr->start_timer("SpSYRK");
            auto C_computed = spsyrk_cyclic_2d<Semiring>(A);
            timer_ptr->stop_timer("SpSYRK");

            double t = timer_ptr->get_timer("SpSYRK");
            
            if (symmetria::my_pe==0) {
                std::cout<<"Time for SpSYRK: "<<t<<"s"<<std::endl;
                std::cout<<"NNZ C: "<<C_computed.get_nnz()<<std::endl;
            }

            /* Write timer to CSV */
            std::string csv_name("timings_spsyrk_2d_"+config.name+"_"+STR(symmetria::my_pe)+".csv");

            /* If first trial, clear the file, else append */
            if (i==0)
                timer_ptr->write_all_timers(csv_name, std::ofstream::trunc);
            else
                timer_ptr->write_all_timers(csv_name, std::ofstream::app);

            timer_ptr->clear_all_timers();
            proc_map->barrier();
        }

    }
    symmetria_finalize();

}

int main(int argc, char ** argv)
{
    std::vector<const char *> req_args {"--rows", "--cols", "--nnz", "--ntrials", "--name", "--type"};

    ExperimentConfig config;
    config.parse_args(argc, argv, req_args);

    std::string type(config.type);

    if (type.compare("2d")==0) {
        run_spsyrk_2d(config);
    }
    if (type.compare("1d")==0) {
        run_spsyrk_1d(config);
    }

    return 0;
}
