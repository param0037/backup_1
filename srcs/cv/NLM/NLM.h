/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/

#ifndef _CUDA_NLM_H_
#define _CUDA_NLM_H_


#include "../cv_classes/cv_classes.h"
#include "../../handles/decx_handles.h"
#include "NLM_BGR.cuh"
#include "NLM_gray.cuh"
#include "NLM_BGR_keep_alpha.cuh"


using de::vis::Img;

namespace de
{
    namespace vis
    {
        namespace cuda
        {
            _DECX_API_ de::DH NLM_RGB(Img& src, Img& dst, uint search_window_radius, uint template_window_radius, float h);


            _DECX_API_ de::DH NLM_RGB_keep_alpha(Img& src, Img& dst, uint search_window_radius, uint template_window_radius, float h);


            _DECX_API_ de::DH NLM_Gray(Img& src, Img& dst, uint search_window_radius, uint template_window_radius, float h);
        }
    }
}



namespace decx
{
    namespace vis
    {
        void NLM_RGB_r16(_Img* src, _Img* dst, uint search_window_radius, uint template_window_radius, float h);


        void NLM_RGB_r16_keep_alpha(_Img* src, _Img* dst, uint search_window_radius, uint template_window_radius, float h);


        void NLM_RGB_r8(_Img* src, _Img* dst, uint search_window_radius, uint template_window_radius, float h);


        void NLM_RGB_r8_keep_alpha(_Img* src, _Img* dst, uint search_window_radius, uint template_window_radius, float h);


        void NLM_gray_r16(_Img* src, _Img* dst, uint search_window_radius, uint template_window_radius, float h);


        void NLM_gray_r8(_Img* src, _Img* dst, uint search_window_radius, uint template_window_radius, float h);
    }
}


void decx::vis::NLM_RGB_r16(_Img* src, _Img* dst, uint search_window_radius, uint template_window_radius, float h)
{
    const uint2 kernel_shift = make_uint2(16 - (search_window_radius + template_window_radius),
        16 - (search_window_radius + template_window_radius));

    const uint eq_ker_len = (search_window_radius * 2 + 1) * (search_window_radius * 2 + 1),
        eq_Wker = search_window_radius * 2 + 1;

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->width, 64) * 64,
        decx::utils::ceil<uint>(dst->height, 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 32, dst_buf_dim.y + 32);

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    checkCudaErrors(cudaMemcpy2DAsync((uchar4*)WS_buffer.ptr + _work_space_dim.x * 16 + 16,
        _work_space_dim.x * sizeof(uchar4),
        src->Mat.ptr, src->pitch * sizeof(uchar4),
        src->width * sizeof(uchar4), src->height, cudaMemcpyHostToDevice, S));

    const dim3 block(16, 16);
    const dim3 grid(dst_buf_dim.y / 16, dst_buf_dim.x / 64);

    switch (template_window_radius)
    {
    case 1:
        cu_NLM_r16_BGR_N3x3 << <grid, block, 0, S >> > (WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 4, dst_buf_dim.x / 4,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    case 2:
        cu_NLM_r16_BGR_N5x5 << <grid, block, 0, S >> > (WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 4, dst_buf_dim.x / 4,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    default:
        break;
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(uchar4),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar4), dst->width * sizeof(uchar4), dst->height,
        cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    decx::alloc::_device_dealloc(&WS_buffer);
    decx::alloc::_device_dealloc(&dst_buffer);
}



void decx::vis::NLM_RGB_r16_keep_alpha(_Img* src, _Img* dst, uint search_window_radius, uint template_window_radius, float h)
{
    const uint2 kernel_shift = make_uint2(16 - (search_window_radius + template_window_radius),
        16 - (search_window_radius + template_window_radius));

    const uint eq_ker_len = (search_window_radius * 2 + 1) * (search_window_radius * 2 + 1),
        eq_Wker = search_window_radius * 2 + 1;

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->width, 64) * 64,
        decx::utils::ceil<uint>(dst->height, 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 32, dst_buf_dim.y + 32);

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    checkCudaErrors(cudaMemcpy2DAsync((uchar4*)WS_buffer.ptr + _work_space_dim.x * 16 + 16,
        _work_space_dim.x * sizeof(uchar4),
        src->Mat.ptr, src->pitch * sizeof(uchar4),
        src->width * sizeof(uchar4), src->height, cudaMemcpyHostToDevice, S));

    const dim3 block(16, 16);
    const dim3 grid(dst_buf_dim.y / 16, dst_buf_dim.x / 64);

    switch (template_window_radius)
    {
    case 1:
        cu_NLM_r16_BGR_KPAL_N3x3 << <grid, block, 0, S >> > (WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 4, dst_buf_dim.x / 4,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    case 2:
        cu_NLM_r16_BGR_KPAL_N5x5 << <grid, block, 0, S >> > (WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 4, dst_buf_dim.x / 4,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    default:
        break;
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(uchar4),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar4), dst->width * sizeof(uchar4), dst->height,
        cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    decx::alloc::_device_dealloc(&WS_buffer);
    decx::alloc::_device_dealloc(&dst_buffer);
}



void decx::vis::NLM_RGB_r8(_Img* src, _Img* dst, uint search_window_radius, uint template_window_radius, float h)
{
    const uint2 kernel_shift = make_uint2(8 - (search_window_radius + template_window_radius),
        8 - (search_window_radius + template_window_radius));

    const uint eq_ker_len = (search_window_radius * 2 + 1) * (search_window_radius * 2 + 1),
        eq_Wker = search_window_radius * 2 + 1;

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->width, 64) * 64,
        decx::utils::ceil<uint>(dst->height, 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 16, dst_buf_dim.y + 16);

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    checkCudaErrors(cudaMemcpy2DAsync((uchar4*)WS_buffer.ptr + _work_space_dim.x * 8 + 8,
        _work_space_dim.x * sizeof(uchar4),
        src->Mat.ptr, src->pitch * sizeof(uchar4),
        src->width * sizeof(uchar4), src->height, cudaMemcpyHostToDevice, S));

    const dim3 block(16, 16);
    const dim3 grid(dst_buf_dim.y / 16, dst_buf_dim.x / 64);

    switch (template_window_radius)
    {
    case 1:
        cu_NLM_r8_BGR_N3x3 << <grid, block, 0, S >> > (WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 4, dst_buf_dim.x / 4,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    case 2:
        cu_NLM_r8_BGR_N5x5 << <grid, block, 0, S >> > (WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 4, dst_buf_dim.x / 4,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    default:
        break;
    }
    
    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(uchar4),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar4), dst->width * sizeof(uchar4), dst->height,
        cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    decx::alloc::_device_dealloc(&WS_buffer);
    decx::alloc::_device_dealloc(&dst_buffer);
}



void decx::vis::NLM_RGB_r8_keep_alpha(_Img* src, _Img* dst, uint search_window_radius, uint template_window_radius, float h)
{
    const uint2 kernel_shift = make_uint2(8 - (search_window_radius + template_window_radius),
        8 - (search_window_radius + template_window_radius));

    const uint eq_ker_len = (search_window_radius * 2 + 1) * (search_window_radius * 2 + 1),
        eq_Wker = search_window_radius * 2 + 1;

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->width, 64) * 64,
        decx::utils::ceil<uint>(dst->height, 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 16, dst_buf_dim.y + 16);

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    checkCudaErrors(cudaMemcpy2DAsync((uchar4*)WS_buffer.ptr + _work_space_dim.x * 8 + 8,
        _work_space_dim.x * sizeof(uchar4),
        src->Mat.ptr, src->pitch * sizeof(uchar4),
        src->width * sizeof(uchar4), src->height, cudaMemcpyHostToDevice, S));

    const dim3 block(16, 16);
    const dim3 grid(dst_buf_dim.y / 16, dst_buf_dim.x / 64);

    switch (template_window_radius)
    {
    case 1:
        cu_NLM_r8_BGR_KPAL_N3x3 << <grid, block, 0, S >> > (WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 4, dst_buf_dim.x / 4,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    case 2:
        cu_NLM_r8_BGR_KPAL_N5x5 << <grid, block, 0, S >> > (WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 4, dst_buf_dim.x / 4,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    default:
        break;
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(uchar4),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar4), dst->width * sizeof(uchar4), dst->height,
        cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    decx::alloc::_device_dealloc(&WS_buffer);
    decx::alloc::_device_dealloc(&dst_buffer);
}




void decx::vis::NLM_gray_r16(_Img* src, _Img* dst, uint search_window_radius, uint template_window_radius, float h)
{
    const uint2 kernel_shift = make_uint2(16 - (search_window_radius + template_window_radius),
        16 - (search_window_radius + template_window_radius));

    const uint eq_ker_len = (search_window_radius * 2 + 1) * (search_window_radius * 2 + 1),
        eq_Wker = search_window_radius * 2 + 1;

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->width, 256) * 256,
        decx::utils::ceil<uint>(dst->height, 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 32, dst_buf_dim.y + 32);

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    checkCudaErrors(cudaMemcpy2DAsync((uchar*)WS_buffer.ptr + _work_space_dim.x * 16 + 16,
        _work_space_dim.x * sizeof(uchar),
        src->Mat.ptr, src->pitch * sizeof(uchar),
        src->width * sizeof(uchar), src->height, cudaMemcpyHostToDevice, S));

    const dim3 block(16, 16);
    const dim3 grid(dst_buf_dim.y / 16, dst_buf_dim.x / 256);

    switch (template_window_radius)
    {
    case 1:
        cu_NLM_r16_gray_N3x3 << <grid, block, 0, S >> > (WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 16, dst_buf_dim.x / 16,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    case 2:
        cu_NLM_r16_gray_N5x5 << <grid, block, 0, S >> > (WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 16, dst_buf_dim.x / 16,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;
        
    default:
        break;
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(uchar),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar), dst->width * sizeof(uchar), dst->height,
        cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    decx::alloc::_device_dealloc(&WS_buffer);
    decx::alloc::_device_dealloc(&dst_buffer);
}



void decx::vis::NLM_gray_r8(_Img* src, _Img* dst, uint search_window_radius, uint template_window_radius, float h)
{
    const uint2 kernel_shift = make_uint2(8 - (search_window_radius + template_window_radius),
        8 - (search_window_radius + template_window_radius));

    const uint eq_ker_len = (search_window_radius * 2 + 1) * (search_window_radius * 2 + 1),
        eq_Wker = search_window_radius * 2 + 1;

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(dst->width, 256) * 256,
        decx::utils::ceil<uint>(dst->height, 16) * 16);

    const uint2 _work_space_dim = make_uint2(dst_buf_dim.x + 16, dst_buf_dim.y + 16);

    decx::PtrInfo<float4> WS_buffer, dst_buffer;
    if (decx::alloc::_device_malloc(&WS_buffer, _work_space_dim.x * _work_space_dim.y * sizeof(uchar))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }
    if (decx::alloc::_device_malloc(&dst_buffer, dst_buf_dim.x * dst_buf_dim.y * sizeof(uchar))) {
        Print_Error_Message(4, ALLOC_FAIL);
        exit(-1);
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    checkCudaErrors(cudaMemcpy2DAsync((uchar*)WS_buffer.ptr + _work_space_dim.x * 8 + 8,
        _work_space_dim.x * sizeof(uchar),
        src->Mat.ptr, src->pitch * sizeof(uchar),
        src->width * sizeof(uchar), src->height, cudaMemcpyHostToDevice, S));

    const dim3 block(16, 16);
    const dim3 grid(dst_buf_dim.y / 16, dst_buf_dim.x / 256);

    switch (template_window_radius)
    {
    case 1:
        cu_NLM_r8_gray_N3x3 << <grid, block, 0, S >> > (WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 16, dst_buf_dim.x / 16,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    case 2:
        cu_NLM_r8_gray_N5x5 << <grid, block, 0, S >> > (WS_buffer.ptr, dst_buffer.ptr, _work_space_dim.x / 16, dst_buf_dim.x / 16,
            eq_ker_len, eq_Wker, kernel_shift, h * h);
        break;

    default:
        break;
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(uchar),
        dst_buffer.ptr, dst_buf_dim.x * sizeof(uchar), dst->width * sizeof(uchar), dst->height,
        cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    decx::alloc::_device_dealloc(&WS_buffer);
    decx::alloc::_device_dealloc(&dst_buffer);
}




de::DH de::vis::cuda::NLM_RGB(Img& src, Img& dst, uint search_window_radius, uint template_window_radius, float h)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        Print_Error_Message(4, NOT_INIT);
        exit(-1);
    }

    decx::_Img* _src = dynamic_cast<decx::_Img*>(&src);
    decx::_Img* _dst = dynamic_cast<decx::_Img*>(&dst);

    if (search_window_radius + template_window_radius > 8) {
        decx::vis::NLM_RGB_r16(_src, _dst, search_window_radius, template_window_radius, h);
    }
    else {
        decx::vis::NLM_RGB_r8(_src, _dst, search_window_radius, template_window_radius, h);
    }
    
    return handle;
}



de::DH de::vis::cuda::NLM_RGB_keep_alpha(Img& src, Img& dst, uint search_window_radius, uint template_window_radius, float h)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        Print_Error_Message(4, NOT_INIT);
        exit(-1);
    }

    decx::_Img* _src = dynamic_cast<decx::_Img*>(&src);
    decx::_Img* _dst = dynamic_cast<decx::_Img*>(&dst);

    if (search_window_radius + template_window_radius > 8) {
        decx::vis::NLM_RGB_r16_keep_alpha(_src, _dst, search_window_radius, template_window_radius, h);
    }
    else {
        decx::vis::NLM_RGB_r8_keep_alpha(_src, _dst, search_window_radius, template_window_radius, h);
    }

    return handle;
}



de::DH de::vis::cuda::NLM_Gray(Img& src, Img& dst, uint search_window_radius, uint template_window_radius, float h)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        Print_Error_Message(4, NOT_INIT);
        exit(-1);
    }

    decx::_Img* _src = dynamic_cast<decx::_Img*>(&src);
    decx::_Img* _dst = dynamic_cast<decx::_Img*>(&dst);

    if (search_window_radius + template_window_radius > 8) {
        decx::vis::NLM_gray_r16(_src, _dst, search_window_radius, template_window_radius, h);
    }
    else {
        decx::vis::NLM_gray_r8(_src, _dst, search_window_radius, template_window_radius, h);
    }

    return handle;
}



#endif
