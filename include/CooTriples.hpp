#ifndef COO_TRIPLES_HPP
#define COO_TRIPLES_HPP

#include "common.h"

namespace symmetria {

template <typename IT, typename DT>
class CooTriples
{

using Triple = std::tuple<IT, IT, DT>;

public:
    
    using iterator = typename std::vector<Triple>::iterator;
    using const_iterator = typename std::vector<Triple>::const_iterator;

    CooTriples(){}

    CooTriples(const std::vector<Triple>& triples):
        triples(triples)
    {
    }


    // Convert from CSR on the host
    CooTriples(const DT * h_vals,
                const IT * h_colinds,
                const IT * h_rowptrs,
                const IT nnz, const IT rows)
    {
        this->csr_to_coo_host(h_vals, h_colinds, h_rowptrs, nnz, rows);
    }


    CooTriples(std::vector<DT> * h_vals,
                std::vector<IT> * h_colinds,
                std::vector<IT> * h_rowptrs)
    {
        this->csr_to_coo_host(h_vals, h_colinds, h_rowptrs);
    }


    CooTriples(Triple * triples, const int64_t n):
        triples(triples, triples+n)
    {
    }


    void csr_to_coo_host(std::vector<DT> * h_vals,
                std::vector<IT> * h_colinds,
                std::vector<IT> * h_rowptrs)
    {
        triples.reserve(h_vals->size());
        for (int j=0; j<h_rowptrs->size()-1; j++) {
            for (int i=h_rowptrs->at(j); i<h_rowptrs->at(j+1); i++) {

                IT col = h_colinds->at(i);
                IT row = j;
                DT val = h_vals->at(i);

                triples.emplace_back(row, col, val);
            }
        }
    }


    void csr_to_coo_host(const DT * h_vals,
                        const IT * h_colinds,
                        const IT * h_rowptrs,
                        const IT nnz, const IT rows)
    {
        triples.reserve(nnz);
        for (int j=0; j<rows; j++)
        {
            for (int i=h_rowptrs[j]; i<h_rowptrs[j+1]; i++)
            {
                triples.emplace_back(j, h_colinds[i], h_vals[i]);
            }
        }
    }


    void add_triples(const Triple * to_add, const IT n)
    {
        for (int i=0; i<n; i++)
        {
            triples.emplace_back(std::get<0>(to_add[i]),
                                 std::get<1>(to_add[i]),
                                 std::get<2>(to_add[i]));
        }
    }


    /* Use std::sort then linear scan */
    template <typename SR>
    void sort_merge_sequential(const SR& semiring)
    {
        assert(triples.size() > 0);

        auto comp = [](Triple& a, Triple& b)
        {
            if (std::get<0>(a) != std::get<0>(b)) {
                return std::get<0>(a) < std::get<0>(b);
            } else {
                return std::get<1>(a) < std::get<1>(b);
            }
        };

        std::sort(triples.begin(), triples.end(),
                    comp);

        std::vector<Triple> merged;
        merged.reserve(triples.size()); //overestimate

        IT prev_row = std::get<0>(triples[0]);
        IT prev_col = std::get<1>(triples[0]);
        DT acc = std::get<2>(triples[0]);
        for (int i=1; i<triples.size(); i++)
        {
            if (prev_col == std::get<1>(triples[i])) {
                //acc = semiring.add(std::get<2>(triples[i]), acc);
                acc += std::get<2>(triples[i]);
            } else {
                merged.emplace_back(prev_row, prev_col, acc);
                prev_row = std::get<0>(triples[i]);
                prev_col = std::get<1>(triples[i]);
                acc = std::get<2>(triples[i]);
            }
        };
        merged.emplace_back(prev_row, prev_col, acc);

        this->triples = merged;

    }



    iterator begin() { return this->triples.begin();}
    iterator end() {return this->triples.end();} 
    const_iterator begin() const { return this->triples.begin();}
    const_iterator end() const {return this->triples.end();} 

    Triple& operator[](size_t idx) {return this->triples[idx];}

    IT size() {return this->triples.size();}
    Triple& at(IT idx) {return this->triples[idx];}


    void rowsort()
    {
        /* Used to sort by rows, break ties with the col index */
        auto row_comp = [](Triple& t1, Triple& t2) 
        { 
            if (std::get<0>(t1) == std::get<0>(t2))
                return (std::get<1>(t1) < std::get<1>(t2));
            else
                return (std::get<0>(t1) < std::get<0>(t2));
        };
        std::sort(triples.begin(), triples.end(), row_comp);
    }


    struct IndHash
    {
        size_t operator()(const std::pair<IT,IT>& inds) const
        {
            return std::hash<IT>{}(inds.first) ^ (std::hash<IT>{}(inds.second)<<1);
        }
    };

    struct IndEquals
    {
        bool operator()(const std::pair<IT,IT>& lhs,
                        const std::pair<IT,IT>& rhs) const
        {
            return lhs.first==rhs.first && lhs.second==rhs.second;
        }
    };


    std::string to_str(const Triple& t1)
    {
        std::stringstream ss;
        ss<<"("<<std::get<0>(t1)<<","<<std::get<1>(t1)<<","<<std::get<2>(t1)<<")";
        return ss.str();
    }


    void dump_to_log(Log * logfile)
    {
        std::for_each(triples.begin(), triples.end(), [=](auto const& t)
            { logfile->OFS()<<to_str(t)<<std::endl; } );
    }


    void dump_to_log(Log * logfile, const char * prefix)
    {
        logfile->OFS()<<prefix<<std::endl;
        std::for_each(triples.begin(), triples.end(), [=](auto const& t)
            { logfile->OFS()<<to_str(t)<<std::endl; } );
    }


    std::vector<Triple> get_triples() {return triples;}
    inline IT get_nnz() {return this->triples.size();}

private:
    std::vector<Triple> triples;
};


}






#endif
