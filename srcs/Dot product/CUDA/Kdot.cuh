/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _KDOT_H_
#define _KDOT_H_


#include "../../basic_process/reductions.cuh"
#include "../../classes/classes_util.h"


__global__
/*
* This kernel function contains multiply-add operation
* @param thr_num : The threads number is half of the total length
*/
void cu_dot_vec4f_start(float4* A, float4* B, const size_t thr_num)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float4 shmem[REDUCTION_VEC4_BLOCK];
    float4 tmp[3];
    tmp[2] = make_float4(0, 0, 0, 0);
    shmem[threadIdx.x] = tmp[2];
    
    if (tid < thr_num) {
        tmp[0] = A[2 * tid];
        tmp[1] = B[2 * tid];
        tmp[2].x = __fmul_rn(tmp[0].x, tmp[1].x);
        tmp[2].y = __fmul_rn(tmp[0].y, tmp[1].y);
        tmp[2].z = __fmul_rn(tmp[0].z, tmp[1].z);
        tmp[2].w = __fmul_rn(tmp[0].w, tmp[1].w);

        tmp[0] = A[2 * tid + 1];
        tmp[1] = B[2 * tid + 1];
        tmp[2].x = fmaf(tmp[0].x, tmp[1].x, tmp[2].x);
        tmp[2].y = fmaf(tmp[0].y, tmp[1].y, tmp[2].y);
        tmp[2].z = fmaf(tmp[0].z, tmp[1].z, tmp[2].z);
        tmp[2].w = fmaf(tmp[0].w, tmp[1].w, tmp[2].w);

        shmem[threadIdx.x] = tmp[2];
    }
    __syncthreads();

    // Take tmp[2] as summing result, tmp[0], tmp[1] as tmps
    int _step = REDUCTION_VEC4_BLOCK / 2;
#pragma unroll 9
    for (int i = 0; i < 9; ++i) {
        if (threadIdx.x < _step) {
            tmp[0] = shmem[threadIdx.x];
            tmp[1] = shmem[threadIdx.x + _step];

            tmp[2].x = __fadd_rn(tmp[0].x, tmp[1].x);
            tmp[2].y = __fadd_rn(tmp[0].y, tmp[1].y);
            tmp[2].z = __fadd_rn(tmp[0].z, tmp[1].z);
            tmp[2].w = __fadd_rn(tmp[0].w, tmp[1].w);
        }
        __syncthreads();
        shmem[threadIdx.x] = tmp[2];
        __syncthreads();
        _step /= 2;
    }
    if (threadIdx.x == 0) {
        B[blockIdx.x] = shmem[0];
    }
}


__global__
/*
* This kernel function contains multiply-add operation, 
* The dot process ends by calling this kernel once
* @param thr_num : The threads number is half of the total length
*/
void cu_dot_vec4f_start_SepDst(float4* A, float4* B, float4* dst, const size_t thr_num)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float4 shmem[REDUCTION_VEC4_BLOCK];
    float4 tmp[3];
    tmp[2] = make_float4(0, 0, 0, 0);
    shmem[threadIdx.x] = tmp[2];

    if (tid < thr_num) {
        tmp[0] = A[2 * tid];
        tmp[1] = B[2 * tid];
        tmp[2].x = __fmul_rn(tmp[0].x, tmp[1].x);
        tmp[2].y = __fmul_rn(tmp[0].y, tmp[1].y);
        tmp[2].z = __fmul_rn(tmp[0].z, tmp[1].z);
        tmp[2].w = __fmul_rn(tmp[0].w, tmp[1].w);

        tmp[0] = A[2 * tid + 1];
        tmp[1] = B[2 * tid + 1];
        tmp[2].x = fmaf(tmp[0].x, tmp[1].x, tmp[2].x);
        tmp[2].y = fmaf(tmp[0].y, tmp[1].y, tmp[2].y);
        tmp[2].z = fmaf(tmp[0].z, tmp[1].z, tmp[2].z);
        tmp[2].w = fmaf(tmp[0].w, tmp[1].w, tmp[2].w);

        shmem[threadIdx.x] = tmp[2];
    }
    __syncthreads();

    // Take tmp[2] as summing result, tmp[0], tmp[1] as tmps
    int _step = REDUCTION_VEC4_BLOCK / 2;
#pragma unroll 9
    for (int i = 0; i < 9; ++i) {
        if (threadIdx.x < _step) {
            tmp[0] = shmem[threadIdx.x];
            tmp[1] = shmem[threadIdx.x + _step];

            tmp[2].x = __fadd_rn(tmp[0].x, tmp[1].x);
            tmp[2].y = __fadd_rn(tmp[0].y, tmp[1].y);
            tmp[2].z = __fadd_rn(tmp[0].z, tmp[1].z);
            tmp[2].w = __fadd_rn(tmp[0].w, tmp[1].w);
        }
        __syncthreads();
        shmem[threadIdx.x] = tmp[2];
        __syncthreads();
        _step /= 2;
    }
    if (threadIdx.x == 0) {
        dst[blockIdx.x] = shmem[0];
    }
}


__global__
/*
* This kernel function contains multiply-add operation
* @param thr_num : The threads number is half of the total length
*/
void cu_dot_vec4f(float4* A, float4* B, const size_t thr_num)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float4 shmem[REDUCTION_VEC4_BLOCK];
    float4 tmp[2];
    tmp[1] = make_float4(0, 0, 0, 0);
    shmem[threadIdx.x] = tmp[1];

    if (tid < thr_num) {
        tmp[0] = A[tid * 2];
        tmp[1] = A[tid * 2 + 1];
        tmp[1].x = __fadd_rn(tmp[0].x, tmp[1].x);
        tmp[1].y = __fadd_rn(tmp[0].y, tmp[1].y);
        tmp[1].z = __fadd_rn(tmp[0].z, tmp[1].z);
        tmp[1].w = __fadd_rn(tmp[0].w, tmp[1].w);

        shmem[threadIdx.x] = tmp[1];
    }

    __syncthreads();

    // Take tmp[2] as summing result, tmp[0], tmp[1] as tmps
    int _step = REDUCTION_VEC4_BLOCK / 2;
#pragma unroll 9
    for (int i = 0; i < 9; ++i) {
        if (threadIdx.x < _step) {
            tmp[0] = shmem[threadIdx.x];
            tmp[1] = shmem[threadIdx.x + _step];

            tmp[1].x = __fadd_rn(tmp[0].x, tmp[1].x);
            tmp[1].y = __fadd_rn(tmp[0].y, tmp[1].y);
            tmp[1].z = __fadd_rn(tmp[0].z, tmp[1].z);
            tmp[1].w = __fadd_rn(tmp[0].w, tmp[1].w);
        }
        __syncthreads();
        shmem[threadIdx.x] = tmp[1];
        __syncthreads();
        _step /= 2;
    }
    if (threadIdx.x == 0) {
        B[blockIdx.x] = shmem[0];
    }
}


namespace decx
{
    void Kdot_fp32(decx::alloc::MIF<float4>* A, decx::alloc::MIF<float4>* B, const size_t dev_len, cudaStream_t* S);


    void dev_Kdot_fp32(float4* dev_A, float4 *dev_B, decx::alloc::MIF<float4>* A, decx::alloc::MIF<float4>* B, 
        const size_t dev_len, cudaStream_t *S);
}



void decx::Kdot_fp32(decx::alloc::MIF<float4>* dev_A, decx::alloc::MIF<float4>* dev_B, const size_t dev_len, cudaStream_t *S)
{
    size_t grid = dev_len / 2, thr_num = dev_len / 2;
    int count = 0;
    while (1) {
        grid = decx::utils::ceil<size_t>(thr_num, REDUCTION_VEC4_BLOCK);

        if (count == 0) {
            if (dev_A->leading) {
                cu_dot_vec4f_start << <grid, REDUCTION_VEC4_BLOCK, 0, *S >> > (dev_A->mem, dev_B->mem, thr_num);
                decx::utils::set_mutex_memory_state<float4, float4>(dev_B, dev_A);
            }
            else {
                cu_dot_vec4f_start << <grid, REDUCTION_VEC4_BLOCK, 0, *S >> > (dev_B->mem, dev_A->mem, thr_num);
                decx::utils::set_mutex_memory_state<float4, float4>(dev_A, dev_B);
            }
        }
        else {
            if (dev_A->leading) {
                cu_dot_vec4f << <grid, REDUCTION_VEC4_BLOCK, 0, *S >> > (dev_A->mem, dev_B->mem, thr_num);
                decx::utils::set_mutex_memory_state<float4, float4>(dev_B, dev_A);
            }
            else {
                cu_dot_vec4f << <grid, REDUCTION_VEC4_BLOCK, 0, *S >> > (dev_B->mem, dev_A->mem, thr_num);
                decx::utils::set_mutex_memory_state<float4, float4>(dev_A, dev_B);
            }
        }
        thr_num = grid / 2;
        ++count;

        if (grid == 1)    break;
    }
}



void decx::dev_Kdot_fp32(float4* dev_A_ori, float4* dev_B_ori, decx::alloc::MIF<float4>* dev_A, decx::alloc::MIF<float4>* dev_B,
    const size_t dev_len, cudaStream_t* S)
{
    size_t grid = dev_len / 2, thr_num = dev_len / 2;
    int count = 0;
    while (1) {
        grid = decx::utils::ceil<size_t>(thr_num, REDUCTION_VEC4_BLOCK);

        if (count == 0) {
            cu_dot_vec4f_start_SepDst << <grid, REDUCTION_VEC4_BLOCK, 0, *S >> > (dev_A_ori, dev_B_ori, dev_A->mem, thr_num);
            decx::utils::set_mutex_memory_state<float4, float4>(dev_A, dev_B);
        }
        else {
            if (dev_A->leading) {
                cu_dot_vec4f << <grid, REDUCTION_VEC4_BLOCK, 0, *S >> > (dev_A->mem, dev_B->mem, thr_num);
                decx::utils::set_mutex_memory_state<float4, float4>(dev_B, dev_A);
            }
            else {
                cu_dot_vec4f << <grid, REDUCTION_VEC4_BLOCK, 0, *S >> > (dev_B->mem, dev_A->mem, thr_num);
                decx::utils::set_mutex_memory_state<float4, float4>(dev_A, dev_B);
            }
        }
        thr_num = grid / 2;
        ++count;

        if (grid == 1)    break;
    }
}


#endif