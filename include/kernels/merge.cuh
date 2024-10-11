#ifndef MERGE_CUH
#define MERGE_CUH

#include "../common.h"
#include "CombBLAS/CombBLAS.h"
#include "../CooTriples.hpp"

using namespace combblas;

namespace symmetria
{


template <typename IT, typename DT>
SpTuples<IT, DT> * make_sptuples(std::tuple<IT, IT, DT> * triples,
                                const IT nnz, const IT rows, const IT cols)
{
    return new SpTuples<IT, DT>((int64_t)nnz, 
                                (int64_t)rows, (int64_t)cols, 
                                triples,
                                true, true);
}


template <typename IT, typename DT>
std::string to_str(const std::tuple<IT, IT, DT> t1)
{
    std::stringstream ss;
    ss<<"("<<std::get<0>(t1)<<","<<std::get<1>(t1)<<","<<std::get<2>(t1)<<")";
    return ss.str();
}


template <typename SR, typename IT, typename DT>
CooTriples<IT, DT> merge_hash_combblas(std::vector<std::tuple<IT, IT, DT> *> to_merge, const IT * nnz_arr, const IT rows, const IT cols)
{
    std::vector<SpTuples<IT, DT>*> sp_tuples_vec;
    sp_tuples_vec.reserve(to_merge.size());

    for (int i=0; i<to_merge.size(); i++)
    {
#ifdef DEBUG_
        for (int j=0; j<nnz_arr[i]; j++)
        {
            logptr->OFS()<<to_str(to_merge[i][j])<<std::endl;
        }
#endif
        auto t = to_merge[i];
        auto sptuples = make_sptuples(t, nnz_arr[i], rows, cols);
        sp_tuples_vec.push_back(sptuples);
    }
    
    auto merged_sptuples = MultiwayMergeHash<SR>(sp_tuples_vec, rows, cols, true, true);

    return CooTriples(merged_sptuples->tuples, merged_sptuples->getnnz());
}


}

#endif
