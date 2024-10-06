#ifndef TEST_DRIVER_HPP
#define TEST_DRIVER_HPP

#include "test_utils.cuh"

#include <nlohmann/json.hpp>

using namespace nlohmann;

namespace symmetria {
namespace testing {

struct TestParams
{
    uint64_t rows, cols, nnz;
    std::string name;

    std::string to_str()
    {
        std::stringstream ss;
        ss<<BRIGHT_YELLOW
             <<"NAME: "<<name<<std::endl
             <<"ROWS: "<<rows<<std::endl
             <<"COLS: "<<cols<<std::endl
             <<"NNZ: "<<nnz<<RESET<<std::endl;
        return ss.str();
    }

};


template <typename DER>
class TestDriver
{

public:

    TestDriver(const std::string json_path, const std::string test_name):
        tests(0), test_name(test_name)
    {
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

    }


    void run_tests()
    {
        int npassed = 0;
        std::stringstream ss;
        for (int i=0; i<tests.size(); i++)
        {
            ss<<BRIGHT_YELLOW<<"=====TEST "<<test_name
                 <<" ["<<i+1<<"/"<<tests.size()<<"]"<<"====="<<RESET
                 <<std::endl
                 <<tests[i].to_str()
                 <<std::endl;
            TEST_PRINT(ss.str());

            ss.str("");

            bool passed = this->run_test(tests[i]);
            if (passed) {
                npassed++;
                TEST_SUCCESS("[" + STR(i+1) + "/" + STR(tests.size()) + "]");
            } else {
                TEST_FAIL();
            }

            ss<<BRIGHT_YELLOW<<"================"<<RESET<<std::endl;
            TEST_PRINT(ss.str());

            ss.str("");
        }

        if (npassed == tests.size()) {
            ss<<GREEN<<"All "<<test_name<<" tests passed!"<<RESET<<std::endl;
            TEST_PRINT(ss.str());
        }
    }


    bool run_test(TestParams& test)
    {
        static_cast<DER*>(this)->run_test_impl(test);
    }


    std::string test_name;
    std::vector<TestParams> tests;

};


}
}






#endif
