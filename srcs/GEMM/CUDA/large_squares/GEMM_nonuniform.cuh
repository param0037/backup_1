/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _CUDA_GEMM_NON_UNIFORM_H_
#define _CUDA_GEMM_NON_UNIFORM_H_

#include "../GEMM_kernel_def.cuh"
#include "../../../classes/core_types.h"
#include "../../../core/defines.h"
#include "../../GEMM_utils.h"
//#include <mma.h>


// last storage (16, 16)
// 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param bounds : ~.x : width (in float4); ~.y : height
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp32_anyWH_specL(float4*                   A,
                              float4*                   B,
                              float4*                   dst,
                              const uint                pitch_A,
                              const uint                pitch_B,
                              const uint                Hdst,
                              const uint                __iter)
{
    uint x_gloA, y_gloA, x_gloB, y_gloB;
    uint x_loc, y_loc;
    
    __shared__ float4 shmemA[32][128 / 8 + 1];
    __shared__ float4 shmemB[32][128 / 8 + 1];
    
    float4 sum[8][2];
    Init_Sum

    float4 tmp_A[2] = { make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0) };
    float4 tmp_B[2] = { make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0) };
    
    x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 4;
    size_t glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) * 2 + blockIdx.y * 32;
    size_t glo_dex_B = y_gloB + pitch_B * x_gloB;
    
    for (uint i = 0; i < __iter; ++i)
    {
        if (x_gloA < Hdst)
            tmp_A[0] = A[glo_dex_A];
        if (x_gloA + 4 < Hdst)
            tmp_A[1] = A[glo_dex_A + pitch_A * 4];
        
        x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

        *((float*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0].x;
        *((float*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[0].y;
        *((float*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[0].z;
        *((float*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[0].w;

        *((float*)&(shmemA[x_loc + 16][threadIdx.x / 16]) + y_loc) = tmp_A[1].x;
        *((float*)&(shmemA[x_loc + 17][threadIdx.x / 16]) + y_loc) = tmp_A[1].y;
        *((float*)&(shmemA[x_loc + 18][threadIdx.x / 16]) + y_loc) = tmp_A[1].z;
        *((float*)&(shmemA[x_loc + 19][threadIdx.x / 16]) + y_loc) = tmp_A[1].w;

        x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

        if (y_gloB < pitch_B)
            tmp_B[0] = B[glo_dex_B];
        if (y_gloB + 1 < pitch_B)
            tmp_B[1] = B[glo_dex_B + 1];
        shmemB[x_loc][y_loc] = tmp_B[0];            //load globalB to shmemB
        shmemB[x_loc + 16][y_loc] = tmp_B[1];
        
        __syncthreads();

        glo_dex_A += 4;
        glo_dex_B += 16 * pitch_B;
        
#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            tmp_A[0] = shmemA[__line][x_loc];
            tmp_A[1] = shmemA[__line + 16][x_loc];

            tmp_B[0] = shmemB[__line][y_loc];
            tmp_B[1] = shmemB[__line + 16][y_loc];

            sfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;

    if (y_gloA * 2 < pitch_B) {
        if (x_gloA < Hdst)          s_store_one_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 2 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 3 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 4 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 5 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 6 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 7 < Hdst)      s_store_one_line(0, glo_dex_A);
    }
}



// last storage (16, 16)
// 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param HB : lenght of linear region(_B.height) (in float)
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp32_ABC_anyWH_specL(float4*                   A,
                                  float4*                   B,
                                  float4*                   C,
                                  float4*                   dst,
                                  const uint                pitch_A,
                                  const uint                pitch_B,
                                  const uint                Hdst,
                                  const uint                __iter)
{
    uint x_gloA, y_gloA, x_gloB, y_gloB;
    uint x_loc, y_loc;
    size_t glo_dex_A, glo_dex_B;

    __shared__ float4 shmemA[32][128 / 8 + 1];
    __shared__ float4 shmemB[32][128 / 8 + 1];

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;

    float4 sum[8][2];
    if (y_gloA * 2 < pitch_B) {
        if (x_gloA < Hdst)              s_loadC_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)          s_loadC_line(1, glo_dex_A);
        if (x_gloA + 2 < Hdst)          s_loadC_line(2, glo_dex_A);
        if (x_gloA + 3 < Hdst)          s_loadC_line(3, glo_dex_A);
        if (x_gloA + 4 < Hdst)          s_loadC_line(4, glo_dex_A);
        if (x_gloA + 5 < Hdst)          s_loadC_line(5, glo_dex_A);
        if (x_gloA + 6 < Hdst)          s_loadC_line(6, glo_dex_A);
        if (x_gloA + 7 < Hdst)          s_loadC_line(7, glo_dex_A);
    }

    float4 tmp_A[2], tmp_B[2];

    x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 4;
    glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) * 2 + blockIdx.y * 32;
    glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        tmp_A[0] = make_float4(0, 0, 0, 0);         tmp_A[1] = make_float4(0, 0, 0, 0);
        tmp_B[0] = make_float4(0, 0, 0, 0);         tmp_B[1] = make_float4(0, 0, 0, 0);

        if (x_gloA < Hdst)
            tmp_A[0] = A[glo_dex_A];
        if (x_gloA + 4 < Hdst)
            tmp_A[1] = A[glo_dex_A + pitch_A * 4];

        x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

        *((float*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0].x;
        *((float*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[0].y;
        *((float*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[0].z;
        *((float*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[0].w;

        *((float*)&(shmemA[x_loc + 16][threadIdx.x / 16]) + y_loc) = tmp_A[1].x;
        *((float*)&(shmemA[x_loc + 17][threadIdx.x / 16]) + y_loc) = tmp_A[1].y;
        *((float*)&(shmemA[x_loc + 18][threadIdx.x / 16]) + y_loc) = tmp_A[1].z;
        *((float*)&(shmemA[x_loc + 19][threadIdx.x / 16]) + y_loc) = tmp_A[1].w;

        x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

        if (y_gloB < pitch_B)
            tmp_B[0] = B[glo_dex_B];
        if (y_gloB + 1 < pitch_B)
            tmp_B[1] = B[glo_dex_B + 1];

        shmemB[x_loc][y_loc] = tmp_B[0];            //load globalB to shmemB
        shmemB[x_loc + 16][y_loc] = tmp_B[1];

        __syncthreads();

        glo_dex_A += 4;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 4;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            tmp_A[0] = shmemA[__line][x_loc];
            tmp_A[1] = shmemA[__line + 16][x_loc];

            tmp_B[0] = shmemB[__line][y_loc];
            tmp_B[1] = shmemB[__line + 16][y_loc];

            sfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;

    if (y_gloA * 2 < pitch_B) {
        if (x_gloA < Hdst)          s_store_one_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 2 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 3 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 4 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 5 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 6 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 7 < Hdst)      s_store_one_line(0, glo_dex_A);
    }
}




// last storage (16, 16)
// 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param HB : lenght of linear region(_B.height) (in float)
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp32_anyWH_anyL(float4*                   A,
                             float4*                   B,
                             float4*                   dst,
                             const uint                pitch_A,
                             const uint                pitch_B,
                             const uint                Hdst,
                             const uint                HB,
                             const uint                __iter)
{
    uint x_gloA, y_gloA, x_gloB, y_gloB;
    uint x_loc, y_loc;
    
    __shared__ float4 shmemA[32][128 / 8 + 1];
    __shared__ float4 shmemB[32][128 / 8 + 1];
    
    float4 sum[8][2];
    Init_Sum

    float4 tmp_A[2], tmp_B[2];
    
    x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 4;
    size_t glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) * 2 + blockIdx.y * 32;
    size_t glo_dex_B = y_gloB + pitch_B * x_gloB;
    
    for (uint i = 0; i < __iter; ++i)
    {
        tmp_A[0] = make_float4(0, 0, 0, 0);         tmp_A[1] = make_float4(0, 0, 0, 0);
        tmp_B[0] = make_float4(0, 0, 0, 0);         tmp_B[1] = make_float4(0, 0, 0, 0);

        if (y_gloA < pitch_A) {
            if (x_gloA < Hdst)
                tmp_A[0] = A[glo_dex_A];
            if (x_gloA + 4 < Hdst)
                tmp_A[1] = A[glo_dex_A + pitch_A * 4];
        }
        
        x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

        *((float*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0].x;
        *((float*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[0].y;
        *((float*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[0].z;
        *((float*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[0].w;

        *((float*)&(shmemA[x_loc + 16][threadIdx.x / 16]) + y_loc) = tmp_A[1].x;
        *((float*)&(shmemA[x_loc + 17][threadIdx.x / 16]) + y_loc) = tmp_A[1].y;
        *((float*)&(shmemA[x_loc + 18][threadIdx.x / 16]) + y_loc) = tmp_A[1].z;
        *((float*)&(shmemA[x_loc + 19][threadIdx.x / 16]) + y_loc) = tmp_A[1].w;

        x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

        if (x_gloB < HB) {
            if (y_gloB < pitch_B)
                tmp_B[0] = B[glo_dex_B];
            if (y_gloB + 1 < pitch_B)
                tmp_B[1] = B[glo_dex_B + 1];
        }

        shmemB[x_loc][y_loc] = tmp_B[0];            //load globalB to shmemB
        shmemB[x_loc + 16][y_loc] = tmp_B[1];
        
        __syncthreads();

        glo_dex_A += 4;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 4;
        x_gloB += 16;
        
#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            tmp_A[0] = shmemA[__line][x_loc];
            tmp_A[1] = shmemA[__line + 16][x_loc];

            tmp_B[0] = shmemB[__line][y_loc];
            tmp_B[1] = shmemB[__line + 16][y_loc];

            sfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;

    if (y_gloA * 2 < pitch_B) {
        if (x_gloA < Hdst)          s_store_one_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 2 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 3 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 4 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 5 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 6 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 7 < Hdst)      s_store_one_line(0, glo_dex_A);
    }
}



// last storage (16, 16)
// 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param HB : lenght of linear region(_B.height) (in float)
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp32_ABC_anyWH_anyL(float4*                   A,
                                 float4*                   B,
                                 float4*                   C,
                                 float4*                   dst,
                                 const uint                pitch_A,
                                 const uint                pitch_B,
                                 const uint                Hdst,
                                 const uint                HB,
                                 const uint                __iter)
{
    uint x_gloA, y_gloA, x_gloB, y_gloB;
    uint x_loc, y_loc;
    size_t glo_dex_A, glo_dex_B;
    
    __shared__ float4 shmemA[32][128 / 8 + 1];
    __shared__ float4 shmemB[32][128 / 8 + 1];
    
    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;
    
    float4 sum[8][2];
    if (y_gloA * 2 < pitch_B) {
        if (x_gloA < Hdst)              s_loadC_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)          s_loadC_line(1, glo_dex_A);
        if (x_gloA + 2 < Hdst)          s_loadC_line(2, glo_dex_A);
        if (x_gloA + 3 < Hdst)          s_loadC_line(3, glo_dex_A);
        if (x_gloA + 4 < Hdst)          s_loadC_line(4, glo_dex_A);
        if (x_gloA + 5 < Hdst)          s_loadC_line(5, glo_dex_A);
        if (x_gloA + 6 < Hdst)          s_loadC_line(6, glo_dex_A);
        if (x_gloA + 7 < Hdst)          s_loadC_line(7, glo_dex_A);
    }
    
    float4 tmp_A[2], tmp_B[2];
    
    x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 4;
    glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) * 2 + blockIdx.y * 32;
    glo_dex_B = y_gloB + pitch_B * x_gloB;
    
    for (uint i = 0; i < __iter; ++i)
    {
        tmp_A[0] = make_float4(0, 0, 0, 0);         tmp_A[1] = make_float4(0, 0, 0, 0);
        tmp_B[0] = make_float4(0, 0, 0, 0);         tmp_B[1] = make_float4(0, 0, 0, 0);

        if (y_gloA < pitch_A) {
            if (x_gloA < Hdst)
                tmp_A[0] = A[glo_dex_A];
            if (x_gloA + 4 < Hdst)
                tmp_A[1] = A[glo_dex_A + pitch_A * 4];
        }
        
        x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

        *((float*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0].x;
        *((float*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[0].y;
        *((float*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[0].z;
        *((float*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[0].w;

        *((float*)&(shmemA[x_loc + 16][threadIdx.x / 16]) + y_loc) = tmp_A[1].x;
        *((float*)&(shmemA[x_loc + 17][threadIdx.x / 16]) + y_loc) = tmp_A[1].y;
        *((float*)&(shmemA[x_loc + 18][threadIdx.x / 16]) + y_loc) = tmp_A[1].z;
        *((float*)&(shmemA[x_loc + 19][threadIdx.x / 16]) + y_loc) = tmp_A[1].w;

        x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

        if (x_gloB < HB) {
            if (y_gloB < pitch_B)
                tmp_B[0] = B[glo_dex_B];
            if (y_gloB + 1 < pitch_B)
                tmp_B[1] = B[glo_dex_B + 1];
        }

        shmemB[x_loc][y_loc] = tmp_B[0];            //load globalB to shmemB
        shmemB[x_loc + 16][y_loc] = tmp_B[1];
        
        __syncthreads();

        glo_dex_A += 4;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 4;
        x_gloB += 16;
        
#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            tmp_A[0] = shmemA[__line][x_loc];
            tmp_A[1] = shmemA[__line + 16][x_loc];

            tmp_B[0] = shmemB[__line][y_loc];
            tmp_B[1] = shmemB[__line + 16][y_loc];

            sfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;

    if (y_gloA * 2 < pitch_B) {
        if (x_gloA < Hdst)          s_store_one_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 2 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 3 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 4 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 5 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 6 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 7 < Hdst)      s_store_one_line(0, glo_dex_A);
    }
}



// last storage (16, 16)
// 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param HB : lenght of linear region(_B.height) (in float)
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp32_specWH_anyL(float4*                   A,
                              float4*                   B,
                              float4*                   dst,
                              const uint                pitch_A,
                              const uint                pitch_B,
                              const uint                HB,
                              const uint                __iter)
{
    uint x_gloA, y_gloA, x_gloB, y_gloB;
    uint x_loc, y_loc;
    
    __shared__ float4 shmemA[32][128 / 8 + 1];
    __shared__ float4 shmemB[32][128 / 8 + 1];
    
    float4 sum[8][2];
    Init_Sum

    float4 tmp_A[2], tmp_B[2];
    
    x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 4;
    size_t glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) * 2 + blockIdx.y * 32;
    size_t glo_dex_B = y_gloB + pitch_B * x_gloB;
    
    for (uint i = 0; i < __iter; ++i)
    {
        tmp_A[0] = make_float4(0, 0, 0, 0);         tmp_A[1] = make_float4(0, 0, 0, 0);
        tmp_B[0] = make_float4(0, 0, 0, 0);         tmp_B[1] = make_float4(0, 0, 0, 0);

        if (y_gloA < pitch_A) {
            tmp_A[0] = A[glo_dex_A];
            tmp_A[1] = A[glo_dex_A + pitch_A * 4];
        }
        
        x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

        *((float*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0].x;
        *((float*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[0].y;
        *((float*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[0].z;
        *((float*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[0].w;

        *((float*)&(shmemA[x_loc + 16][threadIdx.x / 16]) + y_loc) = tmp_A[1].x;
        *((float*)&(shmemA[x_loc + 17][threadIdx.x / 16]) + y_loc) = tmp_A[1].y;
        *((float*)&(shmemA[x_loc + 18][threadIdx.x / 16]) + y_loc) = tmp_A[1].z;
        *((float*)&(shmemA[x_loc + 19][threadIdx.x / 16]) + y_loc) = tmp_A[1].w;

        x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

        if (x_gloB < HB) {
            tmp_B[0] = B[glo_dex_B];
            tmp_B[1] = B[glo_dex_B + 1];
        }

        shmemB[x_loc][y_loc] = tmp_B[0];            //load globalB to shmemB
        shmemB[x_loc + 16][y_loc] = tmp_B[1];
        
        __syncthreads();

        glo_dex_A += 4;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 4;
        x_gloB += 16;
        
#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            tmp_A[0] = shmemA[__line][x_loc];
            tmp_A[1] = shmemA[__line + 16][x_loc];

            tmp_B[0] = shmemB[__line][y_loc];
            tmp_B[1] = shmemB[__line + 16][y_loc];

            sfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;
    s_store(glo_dex_A)
}



// last storage (16, 16)
// 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param HB : lenght of linear region(_B.height) (in float)
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp32_ABC_specWH_anyL(float4*                   A,
                                  float4*                   B,
                                  float4*                   C,
                                  float4*                   dst,
                                  const uint                pitch_A,
                                  const uint                pitch_B,
                                  const uint                HB,
                                  const uint                __iter)
{
    uint x_gloA, y_gloA, x_gloB, y_gloB;
    uint x_loc, y_loc;
    size_t glo_dex_A, glo_dex_B;
    
    __shared__ float4 shmemA[32][128 / 8 + 1];
    __shared__ float4 shmemB[32][128 / 8 + 1];
    
    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;
    
    float4 sum[8][2];
    s_loadC(glo_dex_A);
    
    float4 tmp_A[2], tmp_B[2];
    
    x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 4;
    glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) * 2 + blockIdx.y * 32;
    glo_dex_B = y_gloB + pitch_B * x_gloB;
    
    for (uint i = 0; i < __iter; ++i)
    {
        tmp_A[0] = make_float4(0, 0, 0, 0);         tmp_A[1] = make_float4(0, 0, 0, 0);
        tmp_B[0] = make_float4(0, 0, 0, 0);         tmp_B[1] = make_float4(0, 0, 0, 0);

        if (y_gloA < pitch_A) {
            tmp_A[0] = A[glo_dex_A];
            tmp_A[1] = A[glo_dex_A + pitch_A * 4];
        }
        
        x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

        *((float*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0].x;
        *((float*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[0].y;
        *((float*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[0].z;
        *((float*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[0].w;

        *((float*)&(shmemA[x_loc + 16][threadIdx.x / 16]) + y_loc) = tmp_A[1].x;
        *((float*)&(shmemA[x_loc + 17][threadIdx.x / 16]) + y_loc) = tmp_A[1].y;
        *((float*)&(shmemA[x_loc + 18][threadIdx.x / 16]) + y_loc) = tmp_A[1].z;
        *((float*)&(shmemA[x_loc + 19][threadIdx.x / 16]) + y_loc) = tmp_A[1].w;

        x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

        if (x_gloB < HB) {
            tmp_B[0] = B[glo_dex_B];
            tmp_B[1] = B[glo_dex_B + 1];
        }

        shmemB[x_loc][y_loc] = tmp_B[0];            //load globalB to shmemB
        shmemB[x_loc + 16][y_loc] = tmp_B[1];
        
        __syncthreads();

        glo_dex_A += 4;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 4;
        x_gloB += 16;
        
#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            tmp_A[0] = shmemA[__line][x_loc];
            tmp_A[1] = shmemA[__line + 16][x_loc];

            tmp_B[0] = shmemB[__line][y_loc];
            tmp_B[1] = shmemB[__line + 16][y_loc];

            sfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;
    s_store(glo_dex_A)
}



// -------------------------------------- fp16 -------------------------------------------------


// last storage (16, 16)
// 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param bounds : ~.x : width (in float4); ~.y : height
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp16_anyWH_anyL(float4*                   A,
                             float4*                   B,
                             float4*                   dst,
                             const uint                pitch_A,
                             const uint                pitch_B,
                             const uint                Hdst,
                             const uint                HB,
                             const uint                __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;

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

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    size_t glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    size_t glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&tmp_A) = make_float4(0, 0, 0, 0);

        if (x_gloA < Hdst && y_gloA < pitch_A)
            *((float4*)&tmp_A) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = tmp_A[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = tmp_A[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = tmp_A[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = tmp_A[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (y_gloB < pitch_B && x_gloB < HB)
            *((float4*)&tmp_A) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&tmp_A);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 2;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&tmp_A) = shmemA[__line][x_loc];
            *((float4*)&tmp_B) = shmemB[__line][y_loc];

            hfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    if (y_gloA < pitch_B) {
        if (x_gloA < Hdst)          h_store_one_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)      h_store_one_line(1, glo_dex_A);
        if (x_gloA + 2 < Hdst)      h_store_one_line(2, glo_dex_A);
        if (x_gloA + 3 < Hdst)      h_store_one_line(3, glo_dex_A);
        if (x_gloA + 4 < Hdst)      h_store_one_line(4, glo_dex_A);
        if (x_gloA + 5 < Hdst)      h_store_one_line(5, glo_dex_A);
        if (x_gloA + 6 < Hdst)      h_store_one_line(6, glo_dex_A);
        if (x_gloA + 7 < Hdst)      h_store_one_line(7, glo_dex_A);
    }
#endif
}



// last storage (16, 16)
// 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param bounds : ~.x : width (in float4); ~.y : height
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp16_ABC_anyWH_anyL(float4*                   A,
                                 float4*                   B,
                                 float4*                   C,
                                 float4*                   dst,
                                 const uint                pitch_A,
                                 const uint                pitch_B,
                                 const uint                Hdst,
                                 const uint                HB,
                                 const uint                __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;
    size_t glo_dex_A, glo_dex_B;

    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    half2 sum[8][4];

    if (y_gloA < pitch_B) {
        if (x_gloA < Hdst)           h_loadC_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)       h_loadC_line(1, glo_dex_A);
        if (x_gloA + 2 < Hdst)       h_loadC_line(2, glo_dex_A);
        if (x_gloA + 3 < Hdst)       h_loadC_line(3, glo_dex_A);
        if (x_gloA + 4 < Hdst)       h_loadC_line(4, glo_dex_A);
        if (x_gloA + 5 < Hdst)       h_loadC_line(5, glo_dex_A);
        if (x_gloA + 6 < Hdst)       h_loadC_line(6, glo_dex_A);
        if (x_gloA + 7 < Hdst)       h_loadC_line(7, glo_dex_A);
    }

    half tmp_A[8];
    half2 tmp_B[4];
    half2 fma_tmp;

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&tmp_A) = make_float4(0, 0, 0, 0);

        if (x_gloA < Hdst && y_gloA < pitch_A)
            *((float4*)&tmp_A) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = tmp_A[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = tmp_A[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = tmp_A[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = tmp_A[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (y_gloB < pitch_B && x_gloB < HB)
            *((float4*)&tmp_A) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&tmp_A);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 2;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&tmp_A) = shmemA[__line][x_loc];
            *((float4*)&tmp_B) = shmemB[__line][y_loc];

            hfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    if (y_gloA < pitch_B) {
        if (x_gloA < Hdst)          h_store_one_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)      h_store_one_line(1, glo_dex_A);
        if (x_gloA + 2 < Hdst)      h_store_one_line(2, glo_dex_A);
        if (x_gloA + 3 < Hdst)      h_store_one_line(3, glo_dex_A);
        if (x_gloA + 4 < Hdst)      h_store_one_line(4, glo_dex_A);
        if (x_gloA + 5 < Hdst)      h_store_one_line(5, glo_dex_A);
        if (x_gloA + 6 < Hdst)      h_store_one_line(6, glo_dex_A);
        if (x_gloA + 7 < Hdst)      h_store_one_line(7, glo_dex_A);
    }
#endif
}




// last storage (16, 16)
// 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param bounds : ~.x : width (in float4); ~.y : height
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp16_anyWH_specL(float4*                   A,
                              float4*                   B,
                              float4*                   dst,
                              const uint                pitch_A,
                              const uint                pitch_B,
                              const uint                Hdst,
                              const uint                __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;

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

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    size_t glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    size_t glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&tmp_A) = make_float4(0, 0, 0, 0);

        if (x_gloA < Hdst)
            *((float4*)&tmp_A) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = tmp_A[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = tmp_A[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = tmp_A[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = tmp_A[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (y_gloB < pitch_B)
            *((float4*)&tmp_A) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&tmp_A);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&tmp_A) = shmemA[__line][x_loc];
            *((float4*)&tmp_B) = shmemB[__line][y_loc];

            hfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;
    if (x_gloA < Hdst)          h_store_one_line(0, glo_dex_A);
    if (x_gloA + 1 < Hdst)      h_store_one_line(1, glo_dex_A);
    if (x_gloA + 2 < Hdst)      h_store_one_line(2, glo_dex_A);
    if (x_gloA + 3 < Hdst)      h_store_one_line(3, glo_dex_A);
    if (x_gloA + 4 < Hdst)      h_store_one_line(4, glo_dex_A);
    if (x_gloA + 5 < Hdst)      h_store_one_line(5, glo_dex_A);
    if (x_gloA + 6 < Hdst)      h_store_one_line(6, glo_dex_A);
    if (x_gloA + 7 < Hdst)      h_store_one_line(7, glo_dex_A);
#endif
}


// last storage (16, 16)
// 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param bounds : ~.x : width (in float4); ~.y : height
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp16_ABC_anyWH_specL(float4*                   A,
                                  float4*                   B,
                                  float4*                   C,
                                  float4*                   dst,
                                  const uint                pitch_A,
                                  const uint                pitch_B,
                                  const uint                Hdst,
                                  const uint                __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;
    size_t glo_dex_A, glo_dex_B;

    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    half2 sum[8][4];

    if (y_gloA < pitch_B) {
        if (x_gloA < Hdst)           h_loadC_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)       h_loadC_line(1, glo_dex_A);
        if (x_gloA + 2 < Hdst)       h_loadC_line(2, glo_dex_A);
        if (x_gloA + 3 < Hdst)       h_loadC_line(3, glo_dex_A);
        if (x_gloA + 4 < Hdst)       h_loadC_line(4, glo_dex_A);
        if (x_gloA + 5 < Hdst)       h_loadC_line(5, glo_dex_A);
        if (x_gloA + 6 < Hdst)       h_loadC_line(6, glo_dex_A);
        if (x_gloA + 7 < Hdst)       h_loadC_line(7, glo_dex_A);
    }

    half tmp_A[8];
    half2 tmp_B[4];
    half2 fma_tmp;

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&tmp_A) = make_float4(0, 0, 0, 0);

        if (x_gloA < Hdst)
            *((float4*)&tmp_A) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = tmp_A[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = tmp_A[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = tmp_A[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = tmp_A[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (y_gloB < pitch_B)
            *((float4*)&tmp_A) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&tmp_A);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 2;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&tmp_A) = shmemA[__line][x_loc];
            *((float4*)&tmp_B) = shmemB[__line][y_loc];

            hfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    if (y_gloA < pitch_B) {
        if (x_gloA < Hdst)          h_store_one_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)      h_store_one_line(1, glo_dex_A);
        if (x_gloA + 2 < Hdst)      h_store_one_line(2, glo_dex_A);
        if (x_gloA + 3 < Hdst)      h_store_one_line(3, glo_dex_A);
        if (x_gloA + 4 < Hdst)      h_store_one_line(4, glo_dex_A);
        if (x_gloA + 5 < Hdst)      h_store_one_line(5, glo_dex_A);
        if (x_gloA + 6 < Hdst)      h_store_one_line(6, glo_dex_A);
        if (x_gloA + 7 < Hdst)      h_store_one_line(7, glo_dex_A);
    }
#endif
}



// last storage (16, 16)
// 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param bounds : ~.x : width (in float4); ~.y : height
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp16_specWH_anyL(float4*                   A,
                              float4*                   B,
                              float4*                   dst,
                              const uint                pitch_A,
                              const uint                pitch_B,
                              const uint                HB,
                              const uint                __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;

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

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    size_t glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    size_t glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&tmp_A) = make_float4(0, 0, 0, 0);

        if (y_gloA < pitch_A)
            *((float4*)&tmp_A) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = tmp_A[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = tmp_A[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = tmp_A[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = tmp_A[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (x_gloB < HB)
            *((float4*)&tmp_A) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&tmp_A);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 2;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&tmp_A) = shmemA[__line][x_loc];
            *((float4*)&tmp_B) = shmemB[__line][y_loc];

            hfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;
    h_store(glo_dex_A)
#endif
}


// last storage (16, 16)
// 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param bounds : ~.x : width (in float4); ~.y : height
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp16_ABC_specWH_anyL(float4*                   A,
                                  float4*                   B,
                                  float4*                   C,
                                  float4*                   dst,
                                  const uint                pitch_A,
                                  const uint                pitch_B,
                                  const uint                HB,
                                  const uint                __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;
    size_t glo_dex_A, glo_dex_B;

    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    half2 sum[8][4];
    h_loadC(glo_dex_A);

    half tmp_A[8];
    half2 tmp_B[4];
    half2 fma_tmp;

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&tmp_A) = make_float4(0, 0, 0, 0);

        if (y_gloA < pitch_A)
            *((float4*)&tmp_A) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = tmp_A[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = tmp_A[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = tmp_A[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = tmp_A[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (x_gloB < HB)
            *((float4*)&tmp_A) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&tmp_A);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 2;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&tmp_A) = shmemA[__line][x_loc];
            *((float4*)&tmp_B) = shmemB[__line][y_loc];

            hfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;
    h_store(glo_dex_A);
#endif
}



#endif