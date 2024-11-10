
#ifndef LOG_HPP
#define LOG_HPP

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <tuple>

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) {                                                 \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " ("     \
                  << __FILE__ << ":" << __LINE__ << ")" << std::endl;      \
        exit(err);                                                         \
    }                                                                      \
}


namespace symmetria {


class Log
{
public:

    Log(int rank)
    {
        ofs.open("Logfile"+std::to_string(rank)+".out");
    }

    
    Log(const char * name)
    {
        ofs.open(name);
    }
    

    template <typename T>
    void log_vec(const std::vector<T>& vec, const char * prefix)
    {
        ofs<<prefix<<std::endl;
        std::for_each(vec.begin(), vec.end(), 
                        [this](auto&  elem) {this->ofs<<elem<<'\n';});
        ofs<<std::endl;
    }


    void log_array() {}
    

    template <typename T>
    void log_vec(const std::vector<T>& vec, const char * prefix, const char * suffix)
    {
        ofs<<prefix<<std::endl;
        std::for_each(vec.begin(), vec.end(), 
                        [this](auto&  elem) {this->ofs<<elem<<'\n';});
        ofs<<suffix<<std::endl;
    }


    template <typename T>
    void log_device_array(T * d_arr, size_t n, const char * prefix) 
    {
        if (n==0)  ofs<<prefix<<std::endl;
        std::vector<T> h_arr(n);
        CUDA_CHECK(cudaMemcpy(h_arr.data(), d_arr, sizeof(T)*n, cudaMemcpyDeviceToHost));
        log_vec(h_arr, prefix);
    }


    void newline() {ofs<<std::endl;}

    
    std::ofstream& OFS() {return ofs;}

private:
    std::ofstream ofs;

};



} //inferno

#endif
