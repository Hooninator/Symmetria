#include "Symmetria.hpp"
#include "test_utils.cuh"

using namespace symmetria;

int main(int argc, char ** argv)
{
    symmetria_init();

    if (symmetria::my_pe==0) std::cout<<"Hello world!"<<std::endl;

    TEST_SUCCESS("Hello World");

    symmetria_finalize();

    return 0;
}
