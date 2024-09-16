#ifndef SPSYRK_CUH
#define SPSYRK_CUH

#include "../common.h"
#include "../DistSpMat.hpp"


namespace symmetria {
template <typename IT, typename DT>
DistSpMat1DBlockRow<IT, DT> spsyrk_bulksync_1d_rowblock(DistSpMat1DBlockRow<IT, DT>& A)
{
}
}


#endif
