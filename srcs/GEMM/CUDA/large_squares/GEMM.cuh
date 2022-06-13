/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _GEMM_CUH_
#define _GEMM_CUH_

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
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp32_spec(float4 *                  A,
                       float4 *                  B,
                       float4 *                  dst,
                       const uint                pitch_A,
                       const uint                pitch_B,
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
    glo_dex_A = x_glo * pitch_B + y_glo * 2;
    s_store(glo_dex_A)
}



// -------------------------- fp16 --------------------------------------



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
void cu_GEMM_fp16_spec(float4 *                A,
                       float4 *                B,
                       float4 *                dst,
                       const uint              pitch_A,
                       const uint              pitch_B,
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
    glo_dex_A = x_glo * pitch_B + y_glo;
    h_store(glo_dex_A)
#endif
}



// *********************************************** dst = A * B + c ******************************************************




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
void cu_GEMM_fp32_ABC_spec(float4 *                A,
                           float4 *                B,
                           float4 *                C,
                           float4 *                dst,
                           const uint              pitch_A,
                           const uint              pitch_B,
                           const uint              __iter)
{
    uint x_glo, y_glo;
    size_t glo_dex_A;

    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_glo * pitch_B + y_glo * 2;

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
    glo_dex_A = x_glo * pitch_B + y_glo * 2;
    s_store(glo_dex_A)
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
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp16_ABC_spec(float4 *            A,
                           float4 *            B,
                           float4 *            C,
                           float4 *            dst,
                           const uint          pitch_A,
                           const uint          pitch_B,
                           const uint          __iter)
{
#if __ABOVE_SM_53
    uint x_glo;
    uint y_glo;
    
    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];
    
    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    size_t glo_dex_A = x_glo * pitch_B + y_glo;

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
    glo_dex_A = x_glo * pitch_B + y_glo;
    h_store(glo_dex_A)
#endif
}


#endif