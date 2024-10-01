#ifndef SEMIRINGS_H
#define SEMIRINGS_H


namespace symmetria {
struct PlusTimesSemiring
{
    template <typename T>
    static T add(T a, T b)
    {
        return a + b;
    }

    template <typename T>
    static T multiply(T a, T b)
    {
        return a * b;
    }

};
}


#endif
