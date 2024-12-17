#ifndef EXPERIMENT_COMMON_CUH
#define EXPERIMENT_COMMON_CUH

#include <map>
#include <vector>
#include <algorithm>
#include <exception>

#define EXP_PRINT(msg) do { \
    int rk; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rk); \
    if (rk==0) std::cout<<msg<<std::endl; \
} while(0);


struct ExperimentConfig
{
    unsigned int rows, cols, nnz, ntrials;
    std::string name;
    std::string type;

    void parse_args(int argc, char ** argv, std::vector<const char *>& req_args)
    {
        //TODO: Add tile args

        std::map<std::string, const char *> args;
        for (int i=1; i<argc; i+=2)
        {
            std::string key = std::string(argv[i]);
            const char * val = (argv[i+1]);
            args[key] = val;
        }
        
        /* Check that everyone's there */
        std::for_each(req_args.begin(), req_args.end(), [&](auto const& arg)
            {
                if (args.find(arg)==args.end()) {
                    std::cerr<<"Couldn't find "<<arg<<std::endl;
                    throw std::exception();
                }
            }
        );

        this->nnz= get_arg<unsigned int>("--nnz", args);
        this->rows = get_arg<unsigned int>("--rows", args);
        this->cols= get_arg<unsigned int>("--cols", args);
        this->ntrials= get_arg<unsigned int>("--ntrials", args);

        this->name = get_arg<std::string>("--name", args);
        this->type = get_arg<std::string>("--type", args);
            
    };


    template <typename T>
    T get_arg(const char * s, std::map<std::string, const char *>& args)
    {
        if (args.find(s)==args.end())
        {
            std::cout<<"Couldn't find "<<s<<std::endl;
            if constexpr (std::is_same<T, unsigned int>::value)
            {
                return 0;
            }
            else if constexpr (std::is_same<T, std::string>::value)
            {
                return "null";
            }
        }
        else
        {
            if constexpr (std::is_same<T, unsigned int>::value)
            {
                return std::atoi(args[s]);
            }
            else if constexpr (std::is_same<T, std::string>::value)
            {
                return std::string(args[s]);
            }
        }
    }
};



#endif
