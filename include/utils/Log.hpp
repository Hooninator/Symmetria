
#ifndef LOG_HPP
#define LOG_HPP

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <tuple>


namespace symmetria {


class Log
{
public:

    Log(int rank)
    {
        ofs.open("Logfile"+std::to_string(rank)+".out");
    }

    

    template <typename T>
    void log_vec(std::vector<T>& vec, const char * prefix)
    {
        ofs<<prefix<<std::endl;
        std::for_each(vec.begin(), vec.end(), 
                        [this](auto&  elem) {this->ofs<<elem<<",";});
        ofs<<std::endl;
    }
    

    template <typename T>
    void log_vec(std::vector<T>& vec, const char * prefix, const char * suffix)
    {
        ofs<<prefix<<std::endl;
        std::for_each(vec.begin(), vec.end(), 
                        [this](auto&  elem) {this->ofs<<elem<<'\n';});
        ofs<<suffix<<std::endl;
    }

    
    std::ofstream& OFS() {return ofs;}

private:
    std::ofstream ofs;

};



} //inferno

#endif
