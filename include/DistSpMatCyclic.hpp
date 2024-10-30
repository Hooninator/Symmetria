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


    void build_csr_fast(CooTriples<IT, DT> * triples,
                        std::vector<DT> * h_vals,
                        std::vector<IT> * h_colinds,
                        std::vector<IT> * h_rowptrs,
                        const IT rows)
    {
        std::vector<IT> row_nnz(rows);
        std::for_each(triples->begin(), triples->end(),
            [&](auto const& t)mutable
            {
                row_nnz[std::get<0>(t)]++;
            }
        );

        std::inclusive_scan(row_nnz.begin(), row_nnz.end(), 
                            h_rowptrs->begin()+1);

        std::fill(row_nnz.begin(), row_nnz.end(), IT(0));

        for (int i=0; i<triples->get_nnz(); i++)
        {
            auto const t = triples->at(i);
            const IT rid = std::get<0>(t);
            const IT idx = h_rowptrs->at(rid) + row_nnz[rid];
            h_colinds->at(idx) = std::get<1>(t);
            h_vals->at(idx) = std::get<2>(t);
            row_nnz[rid]++;
        }

    }


    inline void set_rows(const IT _m) {m=_m;}
    inline void set_cols(const IT _n) {n=_n;}
    inline void set_nnz(const IT _nnz) {nnz=_nnz;}

    inline IT get_rows() {return m;}
    inline IT get_cols() {return n;}
    inline IT get_nnz() {return nnz;}

    inline std::vector<SpMat<IT, DT>> get_local_matrices() {return local_matrices;}

    std::shared_ptr<ProcMap> proc_map;

protected:
    IT m, n, nnz;
    std::vector<IT> tile_sizes;

    IT mb, nb;
    IT mtiles, ntiles;
    std::vector<SpMat<IT, DT>> local_matrices;

};


template <typename IT, typename DT>
class DistSpMatCyclic2D : public DistSpMatCyclic<IT, DT>
{
public:

    using Triple = std::tuple<IT, IT, DT>;

    DistSpMatCyclic2D(const IT m, const IT n, const IT nnz,
                    const IT mb, const IT nb, 
                    std::shared_ptr<ProcMap> proc_map):
        DistSpMatCyclic<IT, DT>(m, n, nnz, mb, nb, proc_map)
    {
    }


    Triple map_glob_to_local(const Triple& t)
    {
    }


    Triple map_local_to_tile(const Triple& t)
    {
    }


    int owner(const Triple& t)
    {
        return 0;
    }


    int tile_owner(const Triple& t)
    {
        return 0;
    }


};

}






#endif
