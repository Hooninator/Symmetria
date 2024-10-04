#ifndef TEST_UTILS_CUH
#define TEST_UTILS_CUH

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <nlohmann/json.hpp>

#define RED         "\033[31m"
#define GREEN       "\033[32m"
#define RESET       "\033[0m"

#define TEST_CHECK(condition) \
    do { \
        if (!(condition)) { \
            std::cerr <<RED<< "Error: Assertion failed at " << __FILE__ << ":" << __LINE__ << RESET<<std::endl; \
            std::abort(); \
        } \
    } while (0)

#define TEST_SUCCESS(name) \
    do { \
        MPI_Barrier(MPI_COMM_WORLD); \
        int rank; \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
        if (rank==0) std::cout<<GREEN<<"Test "<<name<<" passed!"<<RESET<<std::endl; \
    } while (0)

#define TEST_PRINT(msg) \
    do { \
        int rank; \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
        if (rank==0) std::cout<<msg<<std::endl; \
    } while (0)

using namespace nlohmann;

namespace symmetria {

namespace testing {

struct TestParams
{
    uint64_t rows, cols, nnz;
    std::string name;
};


std::vector<TestParams> parse_test_json(const std::string json_path)
{

    std::vector<TestParams> tests;

    std::ifstream infile(json_path);

    json json_data;
    infile >> json_data;

    try {
        for (auto const& record : json_data)
        {
            TestParams params;
            params.rows = record.at("rows").get<uint64_t>();
            params.cols = record.at("cols").get<uint64_t>();
            params.nnz = record.at("nnz").get<uint64_t>();
            params.name= record.at("name").get<std::string>();
            tests.push_back(params);
        }
    } catch(json::exception& e) {
        std::cerr<<"Error parsing JSON: "<<e.what() <<std::endl;
    }

    return tests;
}


}


}




#endif
