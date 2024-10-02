#ifndef GALATIC_SPGEMM_CUH
#define GALATIC_SPGEMM_CUH

#include "../common.h"
#include "dCSR.cuh"
#include "SemiRingInterface.h"
#include "../source/device/Multiply.cuh"

#include "transpose_csr.cuh"
#include "dCSR_utils.cuh"


namespace symmetria {

template <typename SR, typename DT, typename IT>
void local_spsyrk(dCSR<DT>& A, dCSR<DT>& A_t, dCSR<DT>& C)
{


}



}

#endif
