/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright © Wayne,
*    2021.04.16
*/


#ifndef _CONV2_MK_IM2COL_H_
#define _CONV2_MK_IM2COL_H_

#include "../../../classes/tensor.h"
#include "../../../classes/TensorArray.h"
#include "im2col.cuh"
#include "eq_GEMM.cuh"
#include "../conv_flags.h"


namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH Conv2_MK_im2col(de::Tensor<float>& src, de::TensorArray<float>& kernel, de::Tensor<float>& dst, const int flag);
    }
}


namespace decx
{
    /**
    * @param __x : the destinated tensor
    * @param dst_dim .x -> width; .y -> height; .z -> depth; .w -> _store_type
    */
    static void sconv_tensor_rearrangement(decx::_Tensor<float>* __x, const uint4 dst_dim);


    static void sconv2_NB_r8x8_mk_im2col(decx::_Tensor<float>* src, decx::_TensorArray<float>* kernel, decx::_Tensor<float>* dst);


    static void sconv2_BC_r8x8_mk_im2col(decx::_Tensor<float>* src, decx::_TensorArray<float>* kernel, decx::_Tensor<float>* dst);
}



static void decx::sconv_tensor_rearrangement(decx::_Tensor<float>* __x, const uint4 dst_dim)
{
    if (__x->width != dst_dim.x || __x->height != dst_dim.y || __x->depth != dst_dim.z) 
    {
        __x->_attribute_assign(dst_dim.x, dst_dim.y, dst_dim.z, dst_dim.w);

        switch (dst_dim.w)
        {
        case decx::DATA_STORE_TYPE::Page_Locked:
            if (decx::alloc::_host_fixed_page_realloc(&__x->Tens, __x->total_bytes)) {
                Print_Error_Message(4, ALLOC_FAIL);
                exit(-1);
            }
            break;

        case decx::DATA_STORE_TYPE::Page_Default:
            if (decx::alloc::_host_virtual_page_realloc(&__x->Tens, __x->total_bytes)) {
                Print_Error_Message(4, ALLOC_FAIL);
                exit(-1);
            }
            break;

        default:
            break;
        }
    }
}



/**
* 
*/
static void decx::sconv2_NB_r8x8_mk_im2col(decx::_Tensor<float>* src, decx::_TensorArray<float>* kernel, decx::_Tensor<float>* dst)
{

    const int2 kernel_shift = make_int2(8 - kernel->height / 2, 8 - kernel->width / 2);
    
    // the width and height of output tensor
    const uint4 dst_o_dim = make_uint4(src->width - (kernel->width / 2) * 2, 
                                       src->height - (kernel->height / 2) * 2, 
                                       kernel->tensor_num,
                                       decx::DATA_STORE_TYPE::Page_Locked);

    decx::sconv_tensor_rearrangement(dst, dst_o_dim);

    // the dimensions of kernel buffer, width : the number of active values in kernel, but dpitch included
    // height : the number of tensors 
    const uint2 ker_buf_dim = make_uint2(decx::utils::ceil<uint>(kernel->plane[0] * (size_t)kernel->dpitch, 128) * 128, 
                                         decx::utils::ceil<uint>(kernel->tensor_num, 32) * 32);

    const uint actual_dst_buf_W = decx::utils::ceil<uint>(kernel->tensor_num, 4) * 4;

    // the dimension of the matrix after im2col operation
    const int2 eq_src_dims = make_int2(decx::utils::ceil<uint>(dst_o_dim.x, 4) * 4, 
                                       dst_o_dim.y);

    // the dimensions of src buffer
    const ulong2 src_buf_dim = make_ulong2((decx::utils::ceil<size_t>(src->width, 64) * 64 + kernel_shift.y * 2) * (size_t)src->dpitch, 
                                            decx::utils::ceil<uint>(src->height, 16) * 16 + kernel_shift.x * 2);

    const ulong2 I2C_dims = make_ulong2(kernel->plane[0] * (size_t)kernel->dpitch, 
                                        eq_src_dims.x * eq_src_dims.y);

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(kernel->tensor_num, 32) * 32, 
                                         decx::utils::ceil<size_t>(I2C_dims.y, 128) * 128);
    
    decx::PtrInfo<float4> src_buf, dst_buf, I2C_buf, ker_buf;

    if (decx::alloc::_device_malloc(&src_buf, src_buf_dim.x * src_buf_dim.y * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&dst_buf, actual_dst_buf_W * dst_buf_dim.y * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&I2C_buf, I2C_dims.x * I2C_dims.y * sizeof(float4) / 4)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&ker_buf, ker_buf_dim.x * ker_buf_dim.y * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    
    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    // copy data from kernel(host) to kernel_buffer(device)
    for (int i = 0; i < kernel->tensor_num; ++i) {
        cudaMemcpy2DAsync(reinterpret_cast<float*>(ker_buf.ptr) + i * ker_buf_dim.x,                            kernel->dpitch * kernel->width * sizeof(float),
                          kernel->TensptrArr.ptr[i],                                                            kernel->dp_x_wp * sizeof(float),
                          kernel->dpitch * kernel->width * sizeof(float),                                        kernel->height,
                          cudaMemcpyHostToDevice,                                                                S);
    }

    // copy data from src(host) to src_buffer(device)
    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(src_buf.ptr) + kernel_shift.x * src_buf_dim.x + kernel_shift.y * src->dpitch,    src_buf_dim.x * sizeof(float), 
        src->Tens.ptr,                                                                                            src->dp_x_wp * sizeof(float),
        src->dp_x_wp * sizeof(float),                                                                            src->height, 
        cudaMemcpyHostToDevice,                                                                                    S));

    int2 thread_bounds = make_int2(eq_src_dims.x / 4, eq_src_dims.y);
    dim3 block_I2C(16, 16);
    dim3 grid_I2C(decx::utils::ceil<int>(thread_bounds.y, 16), decx::utils::ceil<int>(thread_bounds.x, 16));

    cu_sIm2Col_r8_within << <grid_I2C, block_I2C, 0, S >> > (src_buf.ptr,
                                                             I2C_buf.ptr,
                                                             kernel_shift,
                                                             thread_bounds,
                                                             src_buf_dim.x / 4,
                                                             I2C_dims.x / 4,
                                                             make_int2(kernel->width, kernel->height),
                                                             kernel->dpitch / 4);

    const ulong2 MatA_load_bounds = make_ulong2(I2C_dims.x / 4, I2C_dims.y / 4);

    dim3 block_eqMM(32, 8);
    dim3 grid_eqMM(dst_buf_dim.y / 128, dst_buf_dim.x / 32);

    cu_conv_eq_mm << <grid_eqMM, block_eqMM, 0, S >> > (I2C_buf.ptr,
                                                        ker_buf.ptr, 
                                                        dst_buf.ptr, 
                                                        MatA_load_bounds,
                                                        actual_dst_buf_W / 4,
                                                        ker_buf_dim.x / 4,
                                                        ker_buf_dim.x / 128);

    checkCudaErrors(cudaMemcpyAsync(dst->Tens.ptr, dst_buf.ptr, dst->total_bytes, cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&src_buf);
    decx::alloc::_device_dealloc(&dst_buf);
    decx::alloc::_device_dealloc(&ker_buf);
    decx::alloc::_device_dealloc(&I2C_buf);
}




static void decx::sconv2_BC_r8x8_mk_im2col(decx::_Tensor<float>* src, decx::_TensorArray<float>* kernel, decx::_Tensor<float>* dst)
{

    const int2 kernel_shift = make_int2(8 - kernel->height / 2, 8 - kernel->width / 2);
    
    // the width and height of output tensor
    const uint4 dst_o_dim = make_uint4(src->width, 
                                       src->height,
                                       kernel->tensor_num,
                                       decx::DATA_STORE_TYPE::Page_Locked);

    decx::sconv_tensor_rearrangement(dst, dst_o_dim);

    // the dimensions of kernel buffer, width : the number of active values in kernel, but dpitch included
    // height : the number of tensors 
    const uint2 ker_buf_dim = make_uint2(decx::utils::ceil<uint>(kernel->plane[0] * (size_t)kernel->dpitch, 128) * 128, 
                                         decx::utils::ceil<uint>(kernel->tensor_num, 32) * 32);

    const uint actual_dst_buf_W = decx::utils::ceil<uint>(kernel->tensor_num, 4) * 4;

    // the dimension of the matrix after im2col operation
    const int2 eq_src_dims = make_int2(decx::utils::ceil<uint>(dst_o_dim.x, 4) * 4, 
                                       dst_o_dim.y);

    // the dimensions of src buffer
    const ulong2 src_buf_dim = make_ulong2((decx::utils::ceil<size_t>(src->width, 64) * 64 + 16) * (size_t)src->dpitch, 
                                            decx::utils::ceil<uint>(src->height, 16) * 16 + 16);

    const ulong2 I2C_dims = make_ulong2(kernel->plane[0] * (size_t)kernel->dpitch, 
                                        eq_src_dims.x * eq_src_dims.y);

    const uint2 dst_buf_dim = make_uint2(decx::utils::ceil<uint>(kernel->tensor_num, 32) * 32, 
                                         decx::utils::ceil<size_t>(I2C_dims.y, 128) * 128);
    
    decx::PtrInfo<float4> src_buf, dst_buf, I2C_buf, ker_buf;

    if (decx::alloc::_device_malloc(&src_buf, src_buf_dim.x * src_buf_dim.y * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&dst_buf, actual_dst_buf_W * dst_buf_dim.y * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&I2C_buf, I2C_dims.x * I2C_dims.y * sizeof(float4) / 4)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&ker_buf, ker_buf_dim.x * ker_buf_dim.y * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    
    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    // copy data from kernel(host) to kernel_buffer(device)
    for (int i = 0; i < kernel->tensor_num; ++i) {
        cudaMemcpy2DAsync(reinterpret_cast<float*>(ker_buf.ptr) + i * ker_buf_dim.x,            kernel->dpitch * kernel->width * sizeof(float),
                          kernel->TensptrArr.ptr[i],                                            kernel->dp_x_wp * sizeof(float),
                          kernel->dpitch * kernel->width * sizeof(float),                        kernel->height,
                          cudaMemcpyHostToDevice,                                                S);
    }

    // copy data from src(host) to src_buffer(device)
    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<float*>(src_buf.ptr) + 8 * src_buf_dim.x + 8 * src->dpitch,            src_buf_dim.x * sizeof(float), 
        src->Tens.ptr,                                                                            src->dp_x_wp * sizeof(float),
        src->dp_x_wp * sizeof(float),                                                            src->height, 
        cudaMemcpyHostToDevice,                                                                    S));

    int2 thread_bounds = make_int2(eq_src_dims.x / 4, eq_src_dims.y);
    dim3 block_I2C(16, 16);
    dim3 grid_I2C(decx::utils::ceil<int>(thread_bounds.y, 16), decx::utils::ceil<int>(thread_bounds.x, 16));

    cu_sIm2Col_r8_within << <grid_I2C, block_I2C, 0, S >> > (src_buf.ptr,
                                                             I2C_buf.ptr,
                                                             kernel_shift,
                                                             thread_bounds,
                                                             src_buf_dim.x / 4,
                                                             I2C_dims.x / 4,
                                                             make_int2(kernel->width, kernel->height),
                                                             kernel->dpitch / 4);

    const ulong2 MatA_load_bounds = make_ulong2(I2C_dims.x / 4, I2C_dims.y / 4);

    dim3 block_eqMM(32, 8);
    dim3 grid_eqMM(dst_buf_dim.y / 128, dst_buf_dim.x / 32);

    cu_conv_eq_mm << <grid_eqMM, block_eqMM, 0, S >> > (I2C_buf.ptr, 
                                                        ker_buf.ptr, 
                                                        dst_buf.ptr, 
                                                        MatA_load_bounds,
                                                        actual_dst_buf_W / 4,
                                                        ker_buf_dim.x / 4,
                                                        ker_buf_dim.x / 128);

    checkCudaErrors(cudaMemcpyAsync(dst->Tens.ptr, dst_buf.ptr, dst->total_bytes, cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&src_buf);
    decx::alloc::_device_dealloc(&dst_buf);
    decx::alloc::_device_dealloc(&ker_buf);
    decx::alloc::_device_dealloc(&I2C_buf);
}



de::DH de::cuda::Conv2_MK_im2col(de::Tensor<float>& src, de::TensorArray<float>& kernel, de::Tensor<float>& dst, const int flag)
{
    de::DH handle;
    decx::_Tensor<float>* _src = dynamic_cast<decx::_Tensor<float>*>(&src);
    decx::_TensorArray<float>* _kernel = dynamic_cast<decx::_TensorArray<float>*>(&kernel);
    decx::_Tensor<float>* _dst = dynamic_cast<decx::_Tensor<float>*>(&dst);

    if (!decx::cuP.is_init) {
        Print_Error_Message(4, NOT_INIT);
        decx::Not_init(&handle);
        return handle;
    }

    switch (flag) 
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::sconv2_NB_r8x8_mk_im2col(_src, _kernel, _dst);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        decx::sconv2_BC_r8x8_mk_im2col(_src, _kernel, _dst);
        break;

    default:
        break;
    }

    return handle;
}


#endif