#ifndef TILE_WINDOW_HPP
#define TILE_WINDOW_HPP

#include "common.h"
#include "CooTriples.hpp"
#include "SpMat.hpp"

namespace symmetria {

template <typename IT, typename DT>
class TileWindow
{
public:
    TileWindow(uint64_t loc_window_size)
    {
        //NOTE: loc_window_size is assumed to be aligned properly
        MPI_Allreduce(&loc_window_size, &window_size, 1,
                        MPIType<uint64_t>(), MPI_MAX,
                        MPI_COMM_WORLD);

        DEBUG_PRINT("Window size: " + STR(window_size));

        ds_buffer = (char *)(nvshmem_malloc(window_size));
        tip_offset = 0;
    }


    uint64_t add_tile(CooTriples<IT, DT>& triples, const IT m, const IT n)
    {
        local_matrices.emplace_back(m, n, triples, ds_buffer + tip_offset);

        uint64_t result = tip_offset;
        tip_offset += (local_matrices.end()-1)->get_total_bytes();
        assert(tip_offset <= window_size);

        return result;
    }


    void get_tile_sync(char * d_landing_zone, const uint64_t offset, const uint64_t landing_zone_size, 
                        const int target_pe)
    {
        nvshmem_getmem(d_landing_zone, ds_buffer + offset, landing_zone_size, target_pe);
    }


    inline auto get_local_matrices() {return local_matrices;}


    ~TileWindow()
    {
        NVSHMEM_FREE_SAFE(ds_buffer);
    }

private:
    uint64_t window_size;
    uint64_t tip_offset;
    char * ds_buffer;

    std::vector<SpMat<IT, DT>> local_matrices;

};

}




#endif
