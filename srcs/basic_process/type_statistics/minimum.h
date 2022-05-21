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
#include "../../classes/vector.h"
#ifndef GNU_CPUcodes
#include "../reductions.cuh"
#endif


namespace decx
{
    void Min_fp32_vec(decx::_Vector<float>* src, float* res, de::DH* handle);

    void Min_fp32_mat(_Matrix<float>* src, float* res, de::DH* handle);
}


namespace de
{
    _DECX_API_ de::DH Global_Min(de::Vector<float>& src, float* res);

    _DECX_API_ de::DH Global_Min(de::Matrix<float>& src, float* res);
}




void decx::Min_fp32_vec(decx::_Vector<float>* src, float* res, de::DH* handle)
{
    // 2x float4 as a pair of adding operation
    const size_t dev_len = decx::utils::ceil<size_t>(src->length, 8) * 2;

#ifndef GNU_CPUcodes
    decx::PtrInfo<float4> dev_tmp;
    if (decx::alloc::_device_malloc(&dev_tmp, 2 * dev_len * sizeof(float4))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));
    // In comparison, to avoid the buffer len of which number is non-2's power, fill the blanks with one of the elements that ready to be compared
    decx::utils::fill_left_over<float, float4>(reinterpret_cast<float*>(dev_tmp.ptr), dev_tmp.block->block_size / sizeof(float),
        (dev_tmp.block->block_size - dev_len * 4) / sizeof(float), src->Vec.ptr[0], &S);

    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp.ptr;
    dev_B.mem = dev_tmp.ptr + dev_len;

    checkCudaErrors(cudaMemcpyAsync(dev_A.mem, src->Vec.ptr, src->length * sizeof(float), cudaMemcpyHostToDevice, S));
    decx::utils::set_mutex_memory_state<float4, float4>(&dev_A, &dev_B);

    size_t grid = dev_len / 2, thr_num = dev_len / 2;
    while (1) {
        grid = decx::utils::ceil<size_t>(thr_num, REDUCTION_VEC4_BLOCK);
        if (dev_A.leading) {
            cu_min_vec4f << <grid, REDUCTION_VEC4_BLOCK, 0, S >> > (dev_A.mem, dev_B.mem, thr_num, src->Vec.ptr[0]);
            decx::utils::set_mutex_memory_state<float4, float4>(&dev_B, &dev_A);
        }
        else {
            cu_min_vec4f << <grid, REDUCTION_VEC4_BLOCK, 0, S >> > (dev_B.mem, dev_A.mem, thr_num, src->Vec.ptr[0]);
            decx::utils::set_mutex_memory_state<float4, float4>(&dev_A, &dev_B);
        }
        thr_num = grid / 2;

        if (grid == 1)    break;
    }
    float4* ans = new float4();

    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    float _ans = ans->x;
    _ans = GetSmaller(_ans, ans->y);
    _ans = GetSmaller(_ans, ans->z);
    _ans = GetSmaller(_ans, ans->w);
    *res = _ans;
    delete ans;

    decx::alloc::_device_dealloc(&dev_tmp);
    checkCudaErrors(cudaStreamDestroy(S));
#else

#endif
}



void decx::Min_fp32_mat(_Matrix<float>* src, float* res, de::DH* handle)
{
    // 2x float4 as a pair of adding operation
    const size_t dev_len = decx::utils::ceil<size_t>(src->element_num, 8) * 2;

#ifndef GNU_CPUcodes
    decx::PtrInfo<float4> dev_tmp;
    if (decx::alloc::_device_malloc(&dev_tmp,
        (dev_len + decx::utils::ceil<size_t>(dev_len, REDUCTION_VEC4_BLOCK)) * sizeof(float4))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));
    // In comparison, to avoid the buffer len of which number is non-2's power, fill the blanks with one of the elements that ready to be compared
    decx::utils::fill_left_over<float, float4>(reinterpret_cast<float*>(dev_tmp.ptr), dev_tmp.block->block_size / sizeof(float),
        (dev_tmp.block->block_size - dev_len * 4) / sizeof(float), src->Mat.ptr[0], &S);

    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp.ptr;
    dev_B.mem = dev_tmp.ptr + dev_len;

    checkCudaErrors(cudaMemcpy2DAsync(dev_A.mem, src->width * sizeof(float), src->Mat.ptr, src->pitch * sizeof(float),
        src->width * sizeof(float), src->height, cudaMemcpyHostToDevice, S));
    decx::utils::set_mutex_memory_state<float4, float4>(&dev_A, &dev_B);

    size_t grid = dev_len / 2, thr_num = dev_len / 2;
    while (1) {
        grid = decx::utils::ceil<size_t>(thr_num, REDUCTION_VEC4_BLOCK);
        if (dev_A.leading) {
            cu_min_vec4f << <grid, REDUCTION_VEC4_BLOCK, 0, S >> > (dev_A.mem, dev_B.mem, thr_num, src->Mat.ptr[0]);
            decx::utils::set_mutex_memory_state<float4, float4>(&dev_B, &dev_A);
        }
        else {
            cu_min_vec4f << <grid, REDUCTION_VEC4_BLOCK, 0, S >> > (dev_B.mem, dev_A.mem, thr_num, src->Mat.ptr[0]);
            decx::utils::set_mutex_memory_state<float4, float4>(&dev_A, &dev_B);
        }
        thr_num = grid / 2;

        if (grid == 1)    break;
    }
    float4* ans = new float4();

    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    float _ans = ans->x;
    _ans = GetSmaller(_ans, ans->y);
    _ans = GetSmaller(_ans, ans->z);
    _ans = GetSmaller(_ans, ans->w);
    *res = _ans;
    delete ans;

    decx::alloc::_device_dealloc(&dev_tmp);
    checkCudaErrors(cudaStreamDestroy(S));
#else

#endif
}



de::DH de::Global_Min(de::Vector<float>& src, float* res)
{
    decx::_Vector<float>* _src = dynamic_cast<decx::_Vector<float>*>(&src);

    de::DH handle;
    decx::Success(&handle);

    decx::Min_fp32_vec(_src, res, &handle);
    return handle;
}


de::DH de::Global_Min(de::Matrix<float>& src, float* res)
{
    _Matrix<float>* _src = dynamic_cast<_Matrix<float>*>(&src);

    de::DH handle;
    decx::Success(&handle);

    decx::Min_fp32_mat(_src, res, &handle);
    return handle;
}