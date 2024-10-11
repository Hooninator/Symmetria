#ifndef EXPERIMENT_COMMON_CUH
#define EXPERIMENT_COMMON_CUH

#include <map>
#include <vector>
#include <algorithm>
#include <exception>


struct ExperimentConfig
{
    unsigned int rows, cols, nnz, ntrials;
    std::string name;
    std::string type;
};


/* Note: Does not handle ill formed commands well */
ExperimentConfig parse_args(int argc, char ** argv)
{
    std::vector<const char *> req_args {"--rows", "--cols", "--nnz", "--ntrials", "--name", "--type"};

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
        
    return ExperimentConfig {(unsigned int)std::atoi(args["--rows"]), 
                            (unsigned int)std::atoi(args["--cols"]),
                            (unsigned int)std::atoi(args["--nnz"]),
                            (unsigned int)std::atoi(args["--ntrials"]),
                            std::string(args["--name"]),
                            std::string(args["--type"])};
};



#endif
