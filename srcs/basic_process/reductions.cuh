/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once


#include "../core/basic.h"


#define REDUCTION_VEC4_BLOCK 512


__global__
void Fill_vec4(float4* src, const float _ele, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 _fill = make_float4(_ele, _ele, _ele, _ele);
    if (tid < len) {
        src[tid] = _fill;
    }
}



__global__
void Fill_vec2(double2* src, const double _ele, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    double2 _fill = make_double2(_ele, _ele);
    if (tid < len) {
        src[tid] = _fill;
    }
}


namespace decx
{
    namespace utils
    {
        /*
        * @brief : This function is to fill the buffer with an indicated element
        * @param total_len : In _type
        * @param fill : In _type
        */
        template <typename _type, typename _vec_type>
        static void fill_left_over(_type* head_src, const size_t total_len, const size_t _fill, _type _ele, cudaStream_t *S);
    }
}


template <typename _type, typename _vec_type>
static void decx::utils::fill_left_over(_type* head_src, const size_t total_len, const size_t _fill, _type _ele, cudaStream_t *S)
{
    const int vec_len = (sizeof(_vec_type) / sizeof(_type));
    size_t true_fill = decx::utils::ceil(_fill, sizeof(_vec_type) / sizeof(_type));        // in _vec_type
    _vec_type* start_ptr = reinterpret_cast<_vec_type*>(head_src + total_len - true_fill * vec_len);

    switch (vec_len)
    {
    case 4:
        Fill_vec4<<<decx::utils::ceil<size_t>(true_fill, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock, 0, *S>>>(
            reinterpret_cast<float4*>(start_ptr), *((float*)&_ele), true_fill);
        break;

    case 2:
        Fill_vec2 << <decx::utils::ceil<size_t>(true_fill, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock, 0, *S >> > (
            reinterpret_cast<double2*>(start_ptr), *((double*)&_ele), true_fill);
        break;

    default:
        break;
    }
}


__global__
/*
* This kernel function contains multiply-add operation
* @param thr_num : The threads number is half of the total length
*/
void cu_sum_vec4f(float4* A, float4* B, const size_t thr_num)
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





__global__
/*
* This kernel function contains multiply-add operation
* @param thr_num : The threads number is half of the total length
*/
void cu_sum_vec4i(int4* A, int4* B, const size_t thr_num)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ int4 shmem[REDUCTION_VEC4_BLOCK];
    int4 tmp[2];
    tmp[1] = make_int4(0, 0, 0, 0);
    shmem[threadIdx.x] = tmp[1];

    if (tid < thr_num) {
        tmp[0] = A[tid * 2];
        tmp[1] = A[tid * 2 + 1];
        
        tmp[1].x = tmp[0].x + tmp[1].x;
        tmp[1].y = tmp[0].y + tmp[1].y;
        tmp[1].z = tmp[0].z + tmp[1].z;
        tmp[1].w = tmp[0].w + tmp[1].w;

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

            tmp[1].x = tmp[0].x + tmp[1].x;
            tmp[1].y = tmp[0].y + tmp[1].y;
            tmp[1].z = tmp[0].z + tmp[1].z;
            tmp[1].w = tmp[0].w + tmp[1].w;
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



__global__
/*
* This kernel function contains multiply-add operation
* @param thr_num : The threads number is half of the total length
*/
void cu_max_vec4f(float4* A, float4* B, const size_t thr_num, const float _ref)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float4 shmem[REDUCTION_VEC4_BLOCK];
    float4 tmp[2];
    tmp[1] = make_float4(_ref, _ref, _ref, _ref);
    shmem[threadIdx.x] = tmp[1];

    if (tid < thr_num) {
        tmp[0] = A[tid * 2];
        tmp[1] = A[tid * 2 + 1];
        tmp[1].x = GetLarger(tmp[0].x, tmp[1].x);
        tmp[1].y = GetLarger(tmp[0].y, tmp[1].y);
        tmp[1].z = GetLarger(tmp[0].z, tmp[1].z);
        tmp[1].w = GetLarger(tmp[0].w, tmp[1].w);

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

            tmp[1].x = GetLarger(tmp[0].x, tmp[1].x);
            tmp[1].y = GetLarger(tmp[0].y, tmp[1].y);
            tmp[1].z = GetLarger(tmp[0].z, tmp[1].z);
            tmp[1].w = GetLarger(tmp[0].w, tmp[1].w);
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



__global__
/*
* This kernel function contains multiply-add operation
* @param thr_num : The threads number is half of the total length
*/
void cu_min_vec4f(float4* A, float4* B, const size_t thr_num, const float _ref)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float4 shmem[REDUCTION_VEC4_BLOCK];
    float4 tmp[2];
    tmp[1] = make_float4(_ref, _ref, _ref, _ref);
    shmem[threadIdx.x] = tmp[1];

    if (tid < thr_num) {
        tmp[0] = A[tid * 2];
        tmp[1] = A[tid * 2 + 1];
        tmp[1].x = GetSmaller(tmp[0].x, tmp[1].x);
        tmp[1].y = GetSmaller(tmp[0].y, tmp[1].y);
        tmp[1].z = GetSmaller(tmp[0].z, tmp[1].z);
        tmp[1].w = GetSmaller(tmp[0].w, tmp[1].w);

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

            tmp[1].x = GetSmaller(tmp[0].x, tmp[1].x);
            tmp[1].y = GetSmaller(tmp[0].y, tmp[1].y);
            tmp[1].z = GetSmaller(tmp[0].z, tmp[1].z);
            tmp[1].w = GetSmaller(tmp[0].w, tmp[1].w);
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