#ifndef DIST_SP_MAT_CYCLIC_HPP
#define DIST_SP_MAT_CYCLIC_HPP

#include "common.h"
#include "ProcMap.hpp"
#include "CooTriples.hpp"
#include "dCSR.cuh"
#include "dCSR_utils.cuh"
#include "SpMat.hpp"

namespace symmetria
{


template <typename IT, typename DT>
class DistSpMatCyclic
{

public:

    DistSpMatCyclic(){}

    DistSpMatCyclic(const IT m, const IT n, const IT nnz,
                    const IT mb, const IT nb, 
                    std::shared_ptr<ProcMap> proc_map):
        m(m), n(n), nnz(nnz),
        mb(mb), nb(nb),
        mtiles(m / mb), ntiles(n / nb)
    {}


    DistSpMatCyclic(const IT m, const IT n, const IT nnz,
                    const IT mb, const IT nb,
                    std::vector<SpMat<IT, DT>>& local_matrices,
                    std::shared_ptr<ProcMap> proc_map):
        m(m), n(n), nnz(nnz),
        mtiles(m / mb), ntiles(n / nb),
        local_matrices(local_matrices)
    {}


    void set_from_coo(CooTriples<IT, DT> * triples)
    {
    }


protected:
    IT m, n, nnz;
    std::vector<IT> tile_sizes;

    IT mb, nb;
    IT mtiles, ntiles;
    std::vector<SpMat<IT, DT>> local_matrices;

};

}






#endif
