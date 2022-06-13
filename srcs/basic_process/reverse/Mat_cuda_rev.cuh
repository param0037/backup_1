/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once


#include "../../core/basic.h"
#include "../../classes/GPU_matrix.h"


#define _REV2D_THREADDIMY_ 32
#define _REV2D_THREADNUM_ 512
#define _REV2D_SHMEM_OFFSET_ 1

__global__
/*
* blockDim.y = 32, total = 512
* @param proc_dim : .x : width (in float4), .y : height
* @param pitch : true width of the buffer, in float4
*/
void cu_Rev2D_vec4(float4* src, float4* dst, int2 proc_dim, const size_t pitch)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    size_t dex;
    float4 reg;
    __shared__ float4 shmem[16][_REV2D_THREADDIMY_ + _REV2D_SHMEM_OFFSET_];

    if (tidx < proc_dim.y && tidy < proc_dim.x) {
        dex = (size_t)tidx * pitch + (size_t)tidy;
        reg = src[dex];

    }
}




template <typename T>
__global__
/*
* blockDim.y = 32, total = 512
* @param proc_dim : .x : width (in T), .y : height
* @param pitch : true width of the buffer, in T
*/
void cu_Rev2D_s(T* src, T* dst, int2 proc_dim, const size_t pitch)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    size_t dex;
    T reg;

    if (tidx < proc_dim.y && tidy < proc_dim.x) {
        dex = (size_t)tidx * pitch + (size_t)tidy;
        reg = src[dex];
        dex = (size_t)(proc_dim.y - 1 - tidx) * pitch + (size_t)(proc_dim.x - 1 - tidy);
        dst[dex] = reg;
    }
}


namespace de
{
    namespace cuda
    {
        template<typename T>
        de::DH Reverse(de::GPU_Matrix<T>& src, de::GPU_Matrix<T>& dst);
    }
}


namespace decx
{

}


using decx::_GPU_Matrix;

template<typename T>
de::DH de::cuda::Reverse(de::GPU_Matrix<T>& src, de::GPU_Matrix<T>& dst)
{
    _GPU_Matrix<T>* _src = dynamic_cast<_GPU_Matrix<T>*>(&src);
    _GPU_Matrix<T>* _dst = dynamic_cast<_GPU_Matrix<T>*>(&dst);

    de::DH handle;
    decx::Success(&handle);

    dim3 grid(decx::utils::ceil<uint>(_src->height, 16),
        decx::utils::ceil<uint>(_src->width, _REV2D_THREADDIMY_));
    dim3 block(16, _REV2D_THREADDIMY_);

    int2 proc_dim = make_int2(_src->width, _src->height);
    cu_Rev2D_s<T> << <grid, block >> > (_src->Mat.ptr, _dst->Mat.ptr, proc_dim, _src->pitch);

    return handle;
}


template _DECX_API_ de::DH de::cuda::Reverse(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& dst);

template _DECX_API_ de::DH de::cuda::Reverse(de::GPU_Matrix<int>& src, de::GPU_Matrix<int>& dst);
