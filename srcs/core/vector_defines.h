/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#ifndef _VECTOR_DEFINES_H_
#define _VECTOR_DEFINES_H_


// vectors for CPU codes
//#if defined(_DECX_CPU_CODES_) || defined(_DECX_COMBINED_)
#ifndef _DECX_CUDA_CODES_

__align__(8) struct int2
{
    int x, y;
};


__align__(8) struct float2
{
    float x, y;
};


__align__(16) struct float4
{
    float x, y, z, w;
};


__align__(16) struct int4
{
    int x, y, z, w;
};


__align__(16) struct int3
{
    int x, y, z;
};


__align__(4) struct uchar4
{
    uchar x, y, z, w;
};


__align__(16) struct ulong2
{
    unsigned __int64 x, y;
};



static inline int2 make_int2(const int x, const int y) {
    int2 ans;
    ans.x = x;
    ans.y = y;
    return ans;
}



static inline ulong2 make_ulong2(const size_t x, const size_t y) {
    ulong2 ans;
    ans.x = x;
    ans.y = y;
    return ans;
}




static inline int4 make_int4(const int x, const int y, const int z, const int w) {
    int4 ans;
    ans.x = x;
    ans.y = y;
    ans.z = z;
    ans.w = w;
    return ans;
}




static inline int3 make_int3(const int x, const int y, const int z) {
    int3 ans;
    ans.x = x;
    ans.y = y;
    ans.z = z;
    return ans;
}

#endif


static inline uchar4 make_uchar4_from_fp32(const float x, const float y, const float z, const float w)
{
    uchar4 ans;
    ans.x = static_cast<uchar>(x);
    ans.y = static_cast<uchar>(y);
    ans.z = static_cast<uchar>(z);
    ans.w = static_cast<uchar>(w);
    return ans;
}

#endif