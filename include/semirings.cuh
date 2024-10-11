#ifndef SEMIRINGS_CUH
#define SEMIRINGS_CUH

#include "common.h"
#include "SemiRingInterface.h" //From GALATIC



namespace symmetria {
template <typename DT>
struct PlusTimesSemiring : SemiRing<DT, DT, DT>
{

    __host__ __device__ DT multiply(const DT& a, const DT& b) const {return a * b;}
    // Hack to make combblas merging routines compile
    __host__ __device__ static DT add(DT& a, DT& b) {return a + b; } 
    __host__ __device__ DT add(const DT& a, const DT& b) const {return a + b;}
    __host__ __device__ static DT AdditiveIdentity() {return 0;}

};
}


#endif
