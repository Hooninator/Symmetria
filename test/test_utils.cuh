#ifndef TEST_UTILS_CUH
#define TEST_UTILS_CUH

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#define RED         "\033[31m"
#define GREEN       "\033[32m"
#define BRIGHT_YELLOW   "\033[93m"
#define RESET       "\033[0m"


#define TEST_CHECK(condition) \
    do { \
        int eval_condition = condition ? 1 : 0; \
        int rank; \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
        if (!(eval_condition)) { \
            std::cerr <<RED<< "Rank "<<rank<<"; Assertion failed at " << __FILE__ << ":" << __LINE__ << RESET<<std::endl; \
        } \
        MPI_Allreduce(MPI_IN_PLACE, &eval_condition, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);\
        return (bool)(eval_condition); \
    } while (0)


#define TEST_SUCCESS(name) \
    do { \
        MPI_Barrier(MPI_COMM_WORLD); \
        int rank; \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
        if (rank==0) std::cout<<GREEN<<"Test "<<name<<" passed!"<<RESET<<std::endl; \
    } while (0)


#define TEST_FAIL() \
    do { \
        MPI_Barrier(MPI_COMM_WORLD); \
        int rank; \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
        if (rank==0) std::cout<<RED<<"Test failed"<<RESET<<std::endl; \
    } while (0)


#define TEST_PRINT(msg) \
    do { \
        int rank; \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
        if (rank==0) std::cout<<msg<<std::endl; \
    } while (0)




namespace symmetria {
namespace testing {

template<typename Vec1, typename Vec2, typename Comp>
bool compare_vectors(Vec1& v1, Vec2& v2, Comp& comp)
{
    assert(v1.size() == v2.size());
    return std::equal(v1.begin(), v1.end(), v2.begin(), comp);
}

template<typename Vec1, typename Vec2>
bool compare_vectors(Vec1& v1, Vec2& v2)
{
    assert(v1.size() == v2.size());
    return std::equal(v1.begin(), v1.end(), v2.begin());
}


}
}




#endif
