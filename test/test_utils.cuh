#ifndef TEST_UTILS_CUH
#define TEST_UTILS_CUH

#define RED         "\033[31m"
#define GREEN       "\033[32m"
#define RESET       "\033[0m"

#define TEST_CHECK(condition) \
    do { \
        if (!(condition)) { \
            std::cerr <<RED<< "Error: Condition failed at " << __FILE__ << ":" << __LINE__ << RESET<<std::endl; \
            std::abort(); \
        } \
    } while (0)

#define TEST_SUCCESS(name) std::cout<<GREEN<<"Test "<<name<<" passed!"<<RESET<<std::endl;

#define TEST_LOG(msg) std::cout<<msg<<std::endl;

namespace symmetria {

namespace testing {




}


}




#endif
