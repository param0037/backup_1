/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#include "fft_utils.cuh"



__global__
/**
* The whole process is done within shared memory, one thread process 1 elements
*/
void __cu_MixRadix_sort_R_halved(float2*         src,                // src
                                 float4*         dst,                // dst
                                 const int       threads_limit,      // length
                                 const int       bit_len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float4 reg = make_float4(0, 0, 0, 0);       // x, y, z, w

    bool is_in = tid < threads_limit;
    int dst_dex;

    if (is_in) {
        *((float2*)&reg.x) = src[tid];
        reg.z = reg.y;
        reg.y = 0;
        
        GetRev(tid, bit_len,
            (int*)(&Const_Mem[0]) + bit_len,        // 正序基
            (int*)(&Const_Mem[0]) + 2 * bit_len,    // 负序基
            &dst_dex);
        
        dst[dst_dex] = reg;
    }
}




__global__
/**
* The whole process is done within shared memory, one thread process 1 elements
*/
void __cu_MixRadix_sort_R(float*          src,                // src
                          float2*         dst,                // dst
                          const int       threads_limit,      // length
                          const int       bit_len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float2 reg = make_float2(0, 0);       // x, y

    bool is_in = tid < threads_limit;
    int dst_dex;

    if (is_in) {
        reg.x = src[tid];
        
        GetRev(tid, bit_len,
            (int*)(&Const_Mem[0]) + bit_len,        // 正序基
            (int*)(&Const_Mem[0]) + 2 * bit_len,    // 负序基
            &dst_dex);
        
        dst[dst_dex] = reg;
    }
}




__global__
/**
* The whole process is done within shared memory, one thread process 1 elements
*/
void __cu_MixRadix_sort_C_halved(float4*         src,                // src
                                 float4*         dst,                // dst
                                 const int       threads_limit,      // length
                                 const int       bit_len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float4 reg = make_float4(0, 0, 0, 0);       // x, y, z, w

    bool is_in = tid < threads_limit;
    int dst_dex;

    if (is_in) {
        reg = src[tid];
        
        GetRev(tid, bit_len,
            (int*)(&Const_Mem[0]) + bit_len,        // 正序基
            (int*)(&Const_Mem[0]) + 2 * bit_len,    // 负序基
            &dst_dex);

        dst[dst_dex] = reg;
    }
}



__global__
/**
* The whole process is done within shared memory, one thread process 1 elements
*/
void __cu_MixRadix_sort_C(float2*         src,                // src
                          float2*         dst,                // dst
                          const int       threads_limit,      // length
                          const int       bit_len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float2 reg = make_float2(0, 0);       // x, y, z, w

    bool is_in = tid < threads_limit;
    int dst_dex;

    if (is_in) {
        reg = src[tid];
        
        GetRev(tid, bit_len,
            (int*)(&Const_Mem[0]) + bit_len,        // 正序基
            (int*)(&Const_Mem[0]) + 2 * bit_len,    // 负序基
            &dst_dex);

        dst[dst_dex] = reg;
    }
}



// ---------------------------------------------------------------------------------------


__global__
/**
* The whole process is done within shared memory, one thread process 1 elements
* @param Wsrc : The step is float2
*/
void __cu_MixRadix_sort2D_C_halved(float4*         src,                // src
                                   float4*         dst,                // dst
                                   const int       threads_limit,      // length
                                   const int       bit_len,
                                   const int       height_limit,
                                   const int       Wsrc)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;

    float4 reg = make_float4(0, 0, 0, 0);       // x, y, z, w

    bool is_in = tidy < threads_limit && tidx < height_limit;
    int dst_dex;

    if (is_in) {
        reg = src[tidx * Wsrc + tidy];
        
        GetRev(tidy, bit_len,
            (int*)(&Const_Mem[0]) + bit_len,        // 正序基
            (int*)(&Const_Mem[0]) + 2 * bit_len,    // 负序基
            &dst_dex);

        dst[tidx * Wsrc + dst_dex] = reg;
    }
}




__global__
/**
* The whole process is done within shared memory, one thread process 1 elements
* @param Wsrc : The step is float2
*/
void __cu_MixRadix_sort2D_R_halved(float2*         src,                // src
                                   float4*         dst,                // dst
                                   const int       threads_limit,      // length
                                   const int       bit_len,
                                   const int       height_limit,
                                   const int       Wsrc)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;

    float4 reg = make_float4(0, 0, 0, 0);       // x, y, z, w

    bool is_in = tidy < threads_limit && tidx < height_limit;
    int dst_dex;

    if (is_in) {
        *((float2*)&reg) = src[tidx * Wsrc + tidy];
        reg.z = reg.y;
        reg.y = 0;
        
        GetRev(tidy, bit_len,
            (int*)(&Const_Mem[0]) + bit_len,        // 正序基
            (int*)(&Const_Mem[0]) + 2 * bit_len,    // 负序基
            &dst_dex);

        dst[tidx * Wsrc + dst_dex] = reg;
    }
}




__global__
/**
* The whole process is done within shared memory, one thread process 1 elements
* @param Wsrc : The step is float2
*/
void __cu_MixRadix_sort2D_R(float*          src,                // src
                            float2*         dst,                // dst
                            const int       threads_limit,      // length
                            const int       bit_len,
                            const int       height_limit,
                            const int       Wsrc)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;

    float2 reg = make_float2(0, 0);       // x, y, z, w

    bool is_in = tidy < threads_limit && tidx < height_limit;
    int dst_dex;

    if (is_in) {
        reg.x = src[tidx * Wsrc + tidy];
        
        GetRev(tidy, bit_len,
            (int*)(&Const_Mem[0]) + bit_len,        // 正序基
            (int*)(&Const_Mem[0]) + 2 * bit_len,    // 负序基
            &dst_dex);

        dst[tidx * Wsrc + dst_dex] = reg;
    }
}




__global__
/**
* The whole process is done within shared memory, one thread process 1 elements
* @param Wsrc : The step is float2
*/
void __cu_MixRadix_sort2D_C(float2*         src,                // src
                            float2*         dst,                // dst
                            const int       threads_limit,      // length
                            const int       bit_len,
                            const int       height_limit,
                            const int       Wsrc)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;

    float2 reg = make_float2(0, 0);       // x, y, z, w

    bool is_in = tidy < threads_limit && tidx < height_limit;
    int dst_dex;

    if (is_in) {
        reg = src[tidx * Wsrc + tidy];
        
        GetRev(tidy, bit_len,
            (int*)(&Const_Mem[0]) + bit_len,        // 正序基
            (int*)(&Const_Mem[0]) + 2 * bit_len,    // 负序基
            &dst_dex);

        dst[tidx * Wsrc + dst_dex] = reg;
    }
}