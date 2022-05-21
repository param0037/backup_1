/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once


#include "../../core/basic.h"
#include "../../classes/matrix.h"
#ifndef GNU_CPUcodes
#include "transpose.cuh"
#endif


namespace de
{
    _DECX_API_ de::DH Transpose(Matrix<float>& src, Matrix<float>& dst);

    _DECX_API_ de::DH Transpose(Matrix<int>& src, Matrix<int>& dst);

    _DECX_API_ de::DH Transpose(Matrix<double>& src, Matrix<double>& dst);

    _DECX_API_ de::DH Transpose(Matrix<de::Half>& src, Matrix<de::Half>& dst);
}


namespace decx
{
    /**
    * ATTENTION: sizeof(_vec_type) / sizeof(_type) must be 4
    * @param <_type> : This is the typename of the single element
    * @param <_vec_type> : This is the typename of the vector combined with 4 elements
    */
    template <typename _type, typename _vec_type>
    void transpose_4x4(_Matrix<_type>* src, _Matrix<_type>* dst, de::DH* handle);


    /**
    * ATTENTION: sizeof(_vec_type) / sizeof(_type) must be 8
    * @param <_type> : This is the typename of the single element
    * @param <_vec_type> : This is the typename of the vector combined with 8 elements
    */
    template <typename _type, typename _vec_type>
    void transpose_8x8(_Matrix<_type>* src, _Matrix<_type>* dst, de::DH* handle);


    /**
    * ATTENTION: sizeof(_vec_type) / sizeof(_type) must be 2
    * @param <_type> : This is the typename of the single element
    * @param <_vec_type> : This is the typename of the vector combined with 2 elements
    */
    template <typename _type, typename _vec_type>
    void transpose_2x2(_Matrix<_type>* src, _Matrix<_type>* dst, de::DH* handle);
}



template <typename _type, typename _vec_type>
void decx::transpose_8x8(_Matrix<_type>* src, _Matrix<_type>* dst, de::DH* handle)
{
    const int2 dev_dim = make_int2(decx::utils::ceil<int>(src->width, 8),
        decx::utils::ceil<int>(src->height, 8));
    const size_t dev_size = (size_t)dev_dim.x * (size_t)dev_dim.y;

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::PtrInfo<_vec_type> dev_tmp;
    if (decx::alloc::_device_malloc(&dev_tmp, 2 * (8 * dev_size) * sizeof(_vec_type))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    _vec_type* dev_src = dev_tmp.ptr,
        * dev_dst = dev_tmp.ptr + 8 * dev_size;

    checkCudaErrors(cudaMemcpy2DAsync(dev_src, dev_dim.x * sizeof(_vec_type),
        src->Mat.ptr, src->pitch * sizeof(_type), src->width * sizeof(_type), src->height,
        cudaMemcpyHostToDevice, S));

    const dim3 grid(decx::utils::ceil<int>(dev_dim.y, _BLOCK_DEFAULT_),
        decx::utils::ceil<int>(dev_dim.x, _BLOCK_DEFAULT_));
    const dim3 block(_BLOCK_DEFAULT_, _BLOCK_DEFAULT_);

    cu_transpose_vec8x8 << <grid, block, 0, S >> > (dev_src, dev_dst, dev_dim.x, dev_dim.y);

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(_type),
        dev_dst, dev_dim.y * sizeof(_vec_type), dev_dim.y * sizeof(_vec_type), dst->height,
        cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    decx::alloc::_device_dealloc(&dev_tmp);
    checkCudaErrors(cudaStreamDestroy(S));
}



template <typename _type, typename _vec_type>
void decx::transpose_4x4(_Matrix<_type>* src, _Matrix<_type>* dst, de::DH* handle)
{
    const int2 dev_dim = make_int2(decx::utils::ceil<int>(src->width, 4),
        decx::utils::ceil<int>(src->height, 4));
    const size_t dev_size = (size_t)dev_dim.x * (size_t)dev_dim.y;

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::PtrInfo<_vec_type> dev_tmp;
    if (decx::alloc::_device_malloc(&dev_tmp, 2 * 4 * dev_size * sizeof(_vec_type))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    _vec_type *dev_src = dev_tmp.ptr,
              *dev_dst = dev_tmp.ptr + 4 * dev_size;
    
    checkCudaErrors(cudaMemcpy2DAsync(dev_src, dev_dim.x * sizeof(_vec_type),
        src->Mat.ptr, src->pitch * sizeof(_type), src->width * sizeof(_type), src->height,
        cudaMemcpyHostToDevice, S));

    const dim3 grid(decx::utils::ceil<int>(dev_dim.y, _BLOCK_DEFAULT_),
        decx::utils::ceil<int>(dev_dim.x, _BLOCK_DEFAULT_));
    const dim3 block(_BLOCK_DEFAULT_, _BLOCK_DEFAULT_);

    cu_transpose_vec4x4<<<grid, block, 0, S>>>(dev_src, dev_dst, dev_dim.x, dev_dim.y);

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(_type),
        dev_dst, dev_dim.y * sizeof(_vec_type), dev_dim.y * sizeof(_vec_type), dst->height,
        cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    decx::alloc::_device_dealloc(&dev_tmp);
    checkCudaErrors(cudaStreamDestroy(S));
}



template <typename _type, typename _vec_type>
void decx::transpose_2x2(_Matrix<_type>* src, _Matrix<_type>* dst, de::DH* handle)
{
    const int2 dev_dim = make_int2(decx::utils::ceil<int>(src->width, 2),
        decx::utils::ceil<int>(src->height, 2));
    const size_t dev_size = (size_t)dev_dim.x * (size_t)dev_dim.y;

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::PtrInfo<_vec_type> dev_tmp;
    if (decx::alloc::_device_malloc(&dev_tmp, 2 * 2 * dev_size * sizeof(_vec_type))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    _vec_type* dev_src = dev_tmp.ptr,
             * dev_dst = dev_tmp.ptr + 2 * dev_size;

    checkCudaErrors(cudaMemcpy2DAsync(dev_src, dev_dim.x * sizeof(_vec_type),
        src->Mat.ptr, src->pitch * sizeof(_type), src->width * sizeof(_type), src->height,
        cudaMemcpyHostToDevice, S));

    const dim3 grid(decx::utils::ceil<int>(dev_dim.y, _BLOCK_DEFAULT_),
        decx::utils::ceil<int>(dev_dim.x, _BLOCK_DEFAULT_));
    const dim3 block(_BLOCK_DEFAULT_, _BLOCK_DEFAULT_);

    cu_transpose_vec2x2 << <grid, block, 0, S >> > (dev_src, dev_dst, dev_dim.x, dev_dim.y);

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(_type),
        dev_dst, dev_dim.y * sizeof(_vec_type), dev_dim.y * sizeof(_vec_type), dst->height,
        cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    decx::alloc::_device_dealloc(&dev_tmp);
    checkCudaErrors(cudaStreamDestroy(S));
}



de::DH de::Transpose(Matrix<float>& src, Matrix<float>& dst)
{
    _Matrix<float>* _src = dynamic_cast<_Matrix<float>*>(&src);
    _Matrix<float>* _dst = dynamic_cast<_Matrix<float>*>(&dst);

    de::DH handle;
    decx::Success(&handle);
    decx::transpose_4x4<float, float4>(_src, _dst, &handle);

    return handle;
}



de::DH de::Transpose(Matrix<int>& src, Matrix<int>& dst)
{
    _Matrix<int>* _src = dynamic_cast<_Matrix<int>*>(&src);
    _Matrix<int>* _dst = dynamic_cast<_Matrix<int>*>(&dst);

    de::DH handle;
    decx::Success(&handle);
    decx::transpose_4x4<int, int4>(_src, _dst, &handle);

    return handle;
}


de::DH de::Transpose(Matrix<double>& src, Matrix<double>& dst)
{
    _Matrix<double>* _src = dynamic_cast<_Matrix<double>*>(&src);
    _Matrix<double>* _dst = dynamic_cast<_Matrix<double>*>(&dst);

    de::DH handle;
    decx::Success(&handle);
    decx::transpose_2x2<double, double2>(_src, _dst, &handle);

    return handle;
}



de::DH de::Transpose(Matrix<de::Half>& src, Matrix<de::Half>& dst)
{
    _Matrix<de::Half>* _src = dynamic_cast<_Matrix<de::Half>*>(&src);
    _Matrix<de::Half>* _dst = dynamic_cast<_Matrix<de::Half>*>(&dst);

    de::DH handle;
    decx::Success(&handle);
    decx::transpose_8x8<de::Half, float4>(_src, _dst, &handle);

    return handle;
}