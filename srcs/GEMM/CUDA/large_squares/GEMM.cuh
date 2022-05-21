/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#ifndef _GEMM_CUH_
#define _GEMM_CUH_

#include "../../../classes/core_types.h"
#include "../../../core/defines.h"
#include "../../GEMM_utils.h"
#include <mma.h>



/**
* block(16, 16), 每个线程处理 8x8=64 个结果(float)，
* 即一个block处理(128, 128)个float结果, 因此需要每个线程分配64个32-bit寄存器来存储结果
* shmemA -> float4[16][128 / 4]        shmemB -> float4[16][128 / 8]
* 一改前面，用经典矩阵乘，即矩阵B的 __linear 是在列方向, 每一个线程load两个float4, 即8个float
* 
* shared memory 的分布图
* -------                    128
* |        |            -------------------
* |     |            |        B          |
* | A   |    128        |                  |    16
* |     |            -------------------
* |     |
* -------
*/


#define sfma_8x1(name, dex_A, dex_sum) {    \
    sum[dex_sum][0].x = fmaf(tmp_B[0].x, tmp_A[dex_A].name, sum[dex_sum][0].x);        \
    sum[dex_sum][0].y = fmaf(tmp_B[0].y, tmp_A[dex_A].name, sum[dex_sum][0].y);        \
    sum[dex_sum][0].z = fmaf(tmp_B[0].z, tmp_A[dex_A].name, sum[dex_sum][0].z);        \
    sum[dex_sum][0].w = fmaf(tmp_B[0].w, tmp_A[dex_A].name, sum[dex_sum][0].w);        \
    \
    sum[dex_sum][1].x = fmaf(tmp_B[1].x, tmp_A[dex_A].name, sum[dex_sum][1].x);        \
    sum[dex_sum][1].y = fmaf(tmp_B[1].y, tmp_A[dex_A].name, sum[dex_sum][1].y);        \
    sum[dex_sum][1].z = fmaf(tmp_B[1].z, tmp_A[dex_A].name, sum[dex_sum][1].z);        \
    sum[dex_sum][1].w = fmaf(tmp_B[1].w, tmp_A[dex_A].name, sum[dex_sum][1].w);        \
}    \

#define sfma_8x8 {        \
    sfma_8x1(x, 0, 0)    \
    sfma_8x1(y, 0, 1)    \
    sfma_8x1(z, 0, 2)    \
    sfma_8x1(w, 0, 3)    \
\
    sfma_8x1(x, 1, 4)    \
    sfma_8x1(y, 1, 5)    \
    sfma_8x1(z, 1, 6)    \
    sfma_8x1(w, 1, 7)    \
}    \


#define s_store_one_line(dex, _dex_name){    \
    dst[_dex_name] = sum[dex][0];            \
    dst[_dex_name + 1] = sum[dex][1];        \
    _dex_name += pitch_dst;                    \
}    \




#define s_store(dex_name)  {            \
    s_store_one_line(0, dex_name)        \
    s_store_one_line(1, dex_name)        \
    s_store_one_line(2, dex_name)        \
    s_store_one_line(3, dex_name)        \
    s_store_one_line(4, dex_name)        \
    s_store_one_line(5, dex_name)        \
    s_store_one_line(6, dex_name)        \
    s_store_one_line(7, dex_name)        \
}    \



#define _Init_Sum(row){    \
    sum[row][0] = make_float4(0.f, 0.f, 0.f, 0.f);    \
    sum[row][1] = make_float4(0.f, 0.f, 0.f, 0.f);    \
}    \



#define Init_Sum {    \
    _Init_Sum(0);    \
    _Init_Sum(1);    \
    _Init_Sum(2);    \
    _Init_Sum(3);    \
    _Init_Sum(4);    \
    _Init_Sum(5);    \
    _Init_Sum(6);    \
    _Init_Sum(7);    \
}    \





// last storage (16, 16)
// 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_128NT_in(float4 *                  A,
                      float4 *                  B,
                      float4 *                  dst,
                      const uint                pitch_A,
                      const uint                pitch_B,
                      const uint                pitch_dst,
                      const uint                __iter)
{
    uint x_glo;
    uint y_glo;
    
    __shared__ float4 shmemA[32][128 / 8 + 1];
    __shared__ float4 shmemB[32][128 / 8 + 1];
    
    float4 sum[8][2];
    Init_Sum

    float4 tmp_A[2];
    float4 tmp_B[2];
    
    size_t glo_dex_A = ((threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x) * pitch_A + threadIdx.x % 4;
    size_t glo_dex_B = ((threadIdx.x % 16) * 2 + blockIdx.y * 32) + pitch_B * (threadIdx.x / 16);
    
    for (uint i = 0; i < __iter; ++i)
    {
        tmp_A[0] = A[glo_dex_A];
        tmp_A[1] = A[glo_dex_A + pitch_A * 4];
        
        x_glo = 4 * (threadIdx.x % 4);            y_glo = (threadIdx.x % 16) / 4;

        *((float*)&(shmemA[x_glo][threadIdx.x / 16]) + y_glo) = tmp_A[0].x;
        *((float*)&(shmemA[x_glo + 1][threadIdx.x / 16]) + y_glo) = tmp_A[0].y;
        *((float*)&(shmemA[x_glo + 2][threadIdx.x / 16]) + y_glo) = tmp_A[0].z;
        *((float*)&(shmemA[x_glo + 3][threadIdx.x / 16]) + y_glo) = tmp_A[0].w;

        *((float*)&(shmemA[x_glo + 16][threadIdx.x / 16]) + y_glo) = tmp_A[1].x;
        *((float*)&(shmemA[x_glo + 17][threadIdx.x / 16]) + y_glo) = tmp_A[1].y;
        *((float*)&(shmemA[x_glo + 18][threadIdx.x / 16]) + y_glo) = tmp_A[1].z;
        *((float*)&(shmemA[x_glo + 19][threadIdx.x / 16]) + y_glo) = tmp_A[1].w;

        x_glo = threadIdx.x / 16;            y_glo = threadIdx.x % 16;

        tmp_B[0] = B[glo_dex_B];
        tmp_B[1] = B[glo_dex_B + 1];
        shmemB[x_glo][y_glo] = tmp_B[0];            //load globalB to shmemB
        shmemB[x_glo + 16][y_glo] = tmp_B[1];
        
        __syncthreads();

        glo_dex_A += 4;
        glo_dex_B += 16 * pitch_B;
        
#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            tmp_A[0] = shmemA[__line][x_glo];
            tmp_A[1] = shmemA[__line + 16][x_glo];

            tmp_B[0] = shmemB[__line][y_glo];
            tmp_B[1] = shmemB[__line + 16][y_glo];

            sfma_8x8;
        }
        __syncthreads();
    }

    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_glo * pitch_dst + y_glo * 2;
    s_store(glo_dex_A)
}



// -------------------------- fp16 --------------------------------------


#define hfma_8x1(dex_A) {    \
    fma_tmp.x = tmp_A[dex_A];        \
    fma_tmp.y = fma_tmp.x;            \
    sum[dex_A][0] = __hfma2(fma_tmp, tmp_B[0], sum[dex_A][0]);    \
    sum[dex_A][1] = __hfma2(fma_tmp, tmp_B[1], sum[dex_A][1]);    \
    sum[dex_A][2] = __hfma2(fma_tmp, tmp_B[2], sum[dex_A][2]);    \
    sum[dex_A][3] = __hfma2(fma_tmp, tmp_B[3], sum[dex_A][3]);    \
}    \




#define hfma_8x8 {        \
    hfma_8x1(0);        \
    hfma_8x1(1);        \
    hfma_8x1(2);        \
    hfma_8x1(3);        \
    hfma_8x1(4);        \
    hfma_8x1(5);        \
    hfma_8x1(6);        \
    hfma_8x1(7);        \
}    \


#define h_store_one_line(dex, _dex_name){    \
    dst[_dex_name] = *((float4*)&sum[dex][0]);        \
    _dex_name += pitch_dst;    \
}    \



#define h_store(dex_name)  {            \
    h_store_one_line(0, dex_name)        \
    h_store_one_line(1, dex_name)        \
    h_store_one_line(2, dex_name)        \
    h_store_one_line(3, dex_name)        \
    h_store_one_line(4, dex_name)        \
    h_store_one_line(5, dex_name)        \
    h_store_one_line(6, dex_name)        \
    h_store_one_line(7, dex_name)        \
}    \






// last storage (16, 16)
// 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_128NT_fp16_in(float4 *                A,
                           float4 *                B,
                           float4 *                dst,
                           const uint            pitch_A,
                           const uint            pitch_B,
                           const uint            pitch_dst,
                           const uint              __iter)
{
#if __ABOVE_SM_53
    uint x_glo;
    uint y_glo;
    
    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];
    
    half2 sum[8][4];
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        *((float4*)&sum[i][0]) = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    half tmp_A[8];
    half2 tmp_B[4];
    half2 fma_tmp;

    size_t glo_dex_A = (threadIdx.x / 2 + 128 * blockIdx.x) * pitch_A + threadIdx.x % 2;
    size_t glo_dex_B = ((threadIdx.x % 16) + blockIdx.y * 16) + pitch_B * (threadIdx.x / 16);
    
    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&tmp_A) = A[glo_dex_A];
        
        x_glo = 8 * (threadIdx.x % 2);            y_glo = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_glo][threadIdx.x / 16]) + y_glo) = tmp_A[0];
        *((half*)&(shmemA[x_glo + 1][threadIdx.x / 16]) + y_glo) = tmp_A[1];
        *((half*)&(shmemA[x_glo + 2][threadIdx.x / 16]) + y_glo) = tmp_A[2];
        *((half*)&(shmemA[x_glo + 3][threadIdx.x / 16]) + y_glo) = tmp_A[3];
        *((half*)&(shmemA[x_glo + 4][threadIdx.x / 16]) + y_glo) = tmp_A[4];
        *((half*)&(shmemA[x_glo + 5][threadIdx.x / 16]) + y_glo) = tmp_A[5];
        *((half*)&(shmemA[x_glo + 6][threadIdx.x / 16]) + y_glo) = tmp_A[6];
        *((half*)&(shmemA[x_glo + 7][threadIdx.x / 16]) + y_glo) = tmp_A[7];
        
        x_glo = threadIdx.x / 16;                y_glo = threadIdx.x % 16;

        shmemB[x_glo][y_glo] = B[glo_dex_B];            //load globalB to shmemB
        
        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        
#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&tmp_A) = shmemA[__line][x_glo];
            *((float4*)&tmp_B) = shmemB[__line][y_glo];
            
            hfma_8x8;
        }
        __syncthreads();
    }

    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_glo * pitch_dst + y_glo;
    h_store(glo_dex_A)
#endif
}



// *********************************************** dst = A * B + c ******************************************************



#define s_loadC_line(dex, _dex_name){    \
    sum[dex][0] = C[_dex_name];        \
    sum[dex][1] = C[_dex_name + 1];        \
    _dex_name += pitch_dst;    \
}    \




#define s_loadC(dex_name)  {        \
    s_loadC_line(0, dex_name)        \
    s_loadC_line(1, dex_name)        \
    s_loadC_line(2, dex_name)        \
    s_loadC_line(3, dex_name)        \
    s_loadC_line(4, dex_name)        \
    s_loadC_line(5, dex_name)        \
    s_loadC_line(6, dex_name)        \
    s_loadC_line(7, dex_name)        \
}    \



// last storage (16, 16)
// 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_128NT_ABC(float4 *                A,
                       float4 *                B,
                       float4 *                C,
                       float4 *                dst,
                       const uint            pitch_A,
                       const uint            pitch_B,
                       const uint            pitch_dst,
                       const uint              __iter)
{
    uint x_glo, y_glo;
    size_t glo_dex_A;

    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_glo * pitch_dst + y_glo * 2;

    float4 sum[8][2];

    s_loadC(glo_dex_A)            // initialize sum with C

    __shared__ float4 shmemA[32][128 / 8 + 1];
    __shared__ float4 shmemB[32][128 / 8 + 1];
    
    float4 tmp_A[2], tmp_B[2];
    
    glo_dex_A = ((threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x) * pitch_A + threadIdx.x % 4;
    size_t glo_dex_B = ((threadIdx.x % 16) * 2 + blockIdx.y * 32) + pitch_B * (threadIdx.x / 16);
    
    for (uint i = 0; i < __iter; ++i)
    {
        tmp_A[0] = A[glo_dex_A];
        tmp_A[1] = A[glo_dex_A + pitch_A * 4];
        
        x_glo = 4 * (threadIdx.x % 4);            y_glo = (threadIdx.x % 16) / 4;

        *((float*)&(shmemA[x_glo][threadIdx.x / 16]) + y_glo) = tmp_A[0].x;
        *((float*)&(shmemA[x_glo + 1][threadIdx.x / 16]) + y_glo) = tmp_A[0].y;
        *((float*)&(shmemA[x_glo + 2][threadIdx.x / 16]) + y_glo) = tmp_A[0].z;
        *((float*)&(shmemA[x_glo + 3][threadIdx.x / 16]) + y_glo) = tmp_A[0].w;

        *((float*)&(shmemA[x_glo + 16][threadIdx.x / 16]) + y_glo) = tmp_A[1].x;
        *((float*)&(shmemA[x_glo + 17][threadIdx.x / 16]) + y_glo) = tmp_A[1].y;
        *((float*)&(shmemA[x_glo + 18][threadIdx.x / 16]) + y_glo) = tmp_A[1].z;
        *((float*)&(shmemA[x_glo + 19][threadIdx.x / 16]) + y_glo) = tmp_A[1].w;

        x_glo = threadIdx.x / 16;            y_glo = threadIdx.x % 16;

        tmp_B[0] = B[glo_dex_B];
        tmp_B[1] = B[glo_dex_B + 1];
        shmemB[x_glo][y_glo] = tmp_B[0];            //load globalB to shmemB
        shmemB[x_glo + 16][y_glo] = tmp_B[1];
        
        __syncthreads();

        glo_dex_A += 4;
        glo_dex_B += 16 * pitch_B;
        
#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            tmp_A[0] = shmemA[__line][x_glo];
            tmp_A[1] = shmemA[__line + 16][x_glo];

            tmp_B[0] = shmemB[__line][y_glo];
            tmp_B[1] = shmemB[__line + 16][y_glo];

            sfma_8x8;
        }
        __syncthreads();
    }

    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_glo * pitch_dst + y_glo * 2;
    s_store(glo_dex_A)
}





#define h_loadC_line(dex, _dex_name){    \
    *((float4*)&sum[dex][0]) = C[_dex_name];        \
    _dex_name += pitch_dst;    \
}    \



#define h_loadC(dex_name)  {        \
    h_loadC_line(0, dex_name)        \
    h_loadC_line(1, dex_name)        \
    h_loadC_line(2, dex_name)        \
    h_loadC_line(3, dex_name)        \
    h_loadC_line(4, dex_name)        \
    h_loadC_line(5, dex_name)        \
    h_loadC_line(6, dex_name)        \
    h_loadC_line(7, dex_name)        \
}    \



// last storage (16, 16)
// 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_128NT_fp16_ABC(float4 *            A,
                            float4 *            B,
                            float4 *            C,
                            float4 *            dst,
                            const uint            pitch_A,
                            const uint            pitch_B,
                            const uint            pitch_dst,
                            const uint          __iter)
{
#if __ABOVE_SM_53
    uint x_glo;
    uint y_glo;
    
    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];
    
    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    size_t glo_dex_A = x_glo * pitch_dst + y_glo;

    half2 sum[8][4];
    h_loadC(glo_dex_A)

    half tmp_A[8];
    half2 tmp_B[4];
    half2 fma_tmp;

    glo_dex_A = (threadIdx.x / 2 + 128 * blockIdx.x) * pitch_A + threadIdx.x % 2;
    size_t glo_dex_B = ((threadIdx.x % 16) + blockIdx.y * 16) + pitch_B * (threadIdx.x / 16);
    
    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&tmp_A) = A[glo_dex_A];
        
        x_glo = 8 * (threadIdx.x % 2);            y_glo = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_glo][threadIdx.x / 16]) + y_glo) = tmp_A[0];
        *((half*)&(shmemA[x_glo + 1][threadIdx.x / 16]) + y_glo) = tmp_A[1];
        *((half*)&(shmemA[x_glo + 2][threadIdx.x / 16]) + y_glo) = tmp_A[2];
        *((half*)&(shmemA[x_glo + 3][threadIdx.x / 16]) + y_glo) = tmp_A[3];
        *((half*)&(shmemA[x_glo + 4][threadIdx.x / 16]) + y_glo) = tmp_A[4];
        *((half*)&(shmemA[x_glo + 5][threadIdx.x / 16]) + y_glo) = tmp_A[5];
        *((half*)&(shmemA[x_glo + 6][threadIdx.x / 16]) + y_glo) = tmp_A[6];
        *((half*)&(shmemA[x_glo + 7][threadIdx.x / 16]) + y_glo) = tmp_A[7];
        
        x_glo = threadIdx.x / 16;                y_glo = threadIdx.x % 16;

        shmemB[x_glo][y_glo] = B[glo_dex_B];            //load globalB to shmemB
        
        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        
#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&tmp_A) = shmemA[__line][x_glo];
            *((float4*)&tmp_B) = shmemB[__line][y_glo];
            
            hfma_8x8;
        }
        __syncthreads();
    }

    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_glo * pitch_dst + y_glo;
    h_store(glo_dex_A)
#endif
}


#endif