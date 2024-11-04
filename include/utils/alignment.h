#ifndef ALIGNMENT_H
#define ALIGNMENT_H

namespace symmetria {

template <typename T>
uint64_t aligned_offset(uint64_t unaligned_offset)
{
    int alignment = alignof(T);
    return (unaligned_offset + alignment - 1) & ~(alignment - 1);
}



}



#endif
