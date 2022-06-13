/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _DOT_FP32_H_
#define _DOT_FP32_H_


#include "../../../core/basic.h"
#include "../../../classes/Matrix.h"
#include "../../../classes/GPU_matrix.h"
#include "../../../classes/Vector.h"
#include "../../../classes/GPU_Vector.h"
#include "../../../classes/Tensor.h"
#include "../../../classes/GPU_Tensor.h"
#include "../Kdot.cuh"


using decx::_Matrix;
using decx::_GPU_Matrix;
using decx::_Vector;
using decx::_GPU_Vector;
using decx::_Tensor;
using decx::_GPU_Tensor;


namespace decx
{
    void Dot_fp32_vec(decx::_Vector<float>* A, decx::_Vector<float>* B, float* res, de::DH* handle);


    void dev_Dot_fp32_vec(decx::_GPU_Vector<float>* A, decx::_GPU_Vector<float>* B, float* res, de::DH* handle);


    void Dot_fp32_mat(_Matrix<float>* A, _Matrix<float>* B, float* res, de::DH* handle);


    void dev_Dot_fp32_mat(_GPU_Matrix<float>* A, _GPU_Matrix<float>* B, float* res, de::DH* handle);


    void Dot_fp32_ten(_Tensor<float>* A, _Tensor<float>* B, float* res, de::DH* handle);


    void dev_Dot_fp32_ten(_GPU_Tensor<float>* A, _GPU_Tensor<float>* B, float* res, de::DH* handle);
}


namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH Dot(de::Vector<float>& A, de::Vector<float>& B, float* res);


        _DECX_API_ de::DH Dot(de::GPU_Vector<float>& A, de::GPU_Vector<float>& B, float* res);


        _DECX_API_ de::DH Dot(de::Matrix<float>& A, de::Matrix<float>& B, float* res);


        _DECX_API_ de::DH Dot(de::Tensor<float>& A, de::Tensor<float>& B, float* res);


        _DECX_API_ de::DH Dot(de::GPU_Matrix<float>& A, de::GPU_Matrix<float>& B, float* res);


        _DECX_API_ de::DH Dot(de::GPU_Tensor<float>& A, de::GPU_Tensor<float>& B, float* res);
    }
}





void decx::Dot_fp32_vec(decx::_Vector<float>* A, decx::_Vector<float>* B, float* res, de::DH* handle)
{
    // 2x float4 as a pair of adding operation
    const size_t dev_len = decx::utils::ceil<size_t>(A->length, 8) * 2;

    decx::PtrInfo<float4> dev_tmp;
    if (decx::alloc::_device_malloc(&dev_tmp, 2 * dev_len * sizeof(float4))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp.ptr;
    dev_B.mem = dev_tmp.ptr + dev_len;

    checkCudaErrors(cudaMemcpyAsync(dev_A.mem, A->Vec.ptr, A->length * sizeof(float), cudaMemcpyHostToDevice, S));
    checkCudaErrors(cudaMemcpyAsync(dev_B.mem, B->Vec.ptr, B->length * sizeof(float), cudaMemcpyHostToDevice, S));
    decx::utils::set_mutex_memory_state<float4, float4>(&dev_A, &dev_B);
    
    size_t grid = dev_len / 2, thr_num = dev_len / 2;
    int count = 0;
    
    float4* ans = new float4();

    decx::Kdot_fp32(&dev_A, &dev_B, dev_len, &S);
    
    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(ans, dev_B.mem, sizeof(float4), cudaMemcpyDeviceToHost, S));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    float _ans = 0;
    _ans += ans->x;
    _ans += ans->y;
    _ans += ans->z;
    _ans += ans->w;
    *res = _ans;
    delete ans;

    decx::alloc::_device_dealloc(&dev_tmp);
    checkCudaErrors(cudaStreamDestroy(S));
}



void decx::dev_Dot_fp32_vec(decx::_GPU_Vector<float>* A, decx::_GPU_Vector<float>* B, float* res, de::DH* handle)
{
    // 2x float4 as a pair of adding operation
    const size_t dev_len = decx::utils::ceil<size_t>(A->length, 8) * 2;

    decx::PtrInfo<float4> dev_tmp;
    size_t buffer_len = decx::utils::ceil<size_t>(dev_len / 2, REDUCTION_VEC4_BLOCK);
    if (decx::alloc::_device_malloc(&dev_tmp, 2 * buffer_len * sizeof(float4))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp.ptr;
    dev_B.mem = dev_tmp.ptr + buffer_len;

    float4* ans = new float4();

    decx::dev_Kdot_fp32(reinterpret_cast<float4*>(
        A->Vec.ptr), reinterpret_cast<float4*>(B->Vec.ptr), &dev_A, &dev_B, dev_len, &S);

    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(ans, dev_B.mem, sizeof(float4), cudaMemcpyDeviceToHost, S));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    float _ans = 0;
    _ans += ans->x;
    _ans += ans->y;
    _ans += ans->z;
    _ans += ans->w;
    *res = _ans;
    delete ans;

    decx::alloc::_device_dealloc(&dev_tmp);
    checkCudaErrors(cudaStreamDestroy(S));
}



void decx::Dot_fp32_mat(_Matrix<float>* A, _Matrix<float>* B, float* res, de::DH* handle)
{
    // 2x float4 as a pair of adding operation
    const size_t dev_len = decx::utils::ceil<size_t>(A->element_num, 8) * 2;

    decx::PtrInfo<float4> dev_tmp;
    if (decx::alloc::_device_malloc(&dev_tmp, 2 * dev_len * sizeof(float4))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp.ptr;
    dev_B.mem = dev_tmp.ptr + dev_len;

    checkCudaErrors(cudaMemcpy2DAsync(dev_A.mem, A->width * sizeof(float), A->Mat.ptr, A->pitch * sizeof(float),
        A->width * sizeof(float), A->height, cudaMemcpyHostToDevice, S));
    checkCudaErrors(cudaMemcpy2DAsync(dev_B.mem, B->width * sizeof(float), B->Mat.ptr, B->pitch * sizeof(float),
        B->width * sizeof(float), B->height, cudaMemcpyHostToDevice, S));
    decx::utils::set_mutex_memory_state<float4, float4>(&dev_A, &dev_B);

    float4* ans = new float4();

    decx::Kdot_fp32(&dev_A, &dev_B, dev_len, &S);

    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(ans, dev_B.mem, sizeof(float4), cudaMemcpyDeviceToHost, S));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    float _ans = 0;
    _ans += ans->x;
    _ans += ans->y;
    _ans += ans->z;
    _ans += ans->w;
    *res = _ans;
    delete ans;

    decx::alloc::_device_dealloc(&dev_tmp);
    checkCudaErrors(cudaStreamDestroy(S));
}



void decx::dev_Dot_fp32_mat(_GPU_Matrix<float>* A, _GPU_Matrix<float>* B, float* res, de::DH* handle)
{
    // 2x float4 as a pair of adding operation
    const size_t dev_len = decx::utils::ceil<size_t>(A->_element_num, 8) * 2;

    decx::PtrInfo<float4> dev_tmp;
    size_t buffer_len = decx::utils::ceil<size_t>(dev_len / 2, REDUCTION_VEC4_BLOCK);
    if (decx::alloc::_device_malloc(&dev_tmp, 2 * buffer_len * sizeof(float4))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp.ptr;
    dev_B.mem = dev_tmp.ptr + buffer_len;

    float4* ans = new float4();

    decx::dev_Kdot_fp32(reinterpret_cast<float4*>(
        A->Mat.ptr), reinterpret_cast<float4*>(B->Mat.ptr), &dev_A, &dev_B, dev_len, &S);

    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(ans, dev_B.mem, sizeof(float4), cudaMemcpyDeviceToHost, S));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    float _ans = 0;
    _ans += ans->x;
    _ans += ans->y;
    _ans += ans->z;
    _ans += ans->w;
    *res = _ans;
    delete ans;

    decx::alloc::_device_dealloc(&dev_tmp);
    checkCudaErrors(cudaStreamDestroy(S));
}



void decx::dev_Dot_fp32_ten(_GPU_Tensor<float>* A, _GPU_Tensor<float>* B, float* res, de::DH* handle)
{
    // 2x float4 as a pair of adding operation
    const size_t dev_len = decx::utils::ceil<size_t>(A->_element_num, 8) * 2;

    decx::PtrInfo<float4> dev_tmp;
    size_t buffer_len = decx::utils::ceil<size_t>(dev_len / 2, REDUCTION_VEC4_BLOCK);
    if (decx::alloc::_device_malloc(&dev_tmp, 2 * buffer_len * sizeof(float4))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp.ptr;
    dev_B.mem = dev_tmp.ptr + buffer_len;

    float4* ans = new float4();

    decx::dev_Kdot_fp32(reinterpret_cast<float4*>(
        A->Tens.ptr), reinterpret_cast<float4*>(B->Tens.ptr), &dev_A, &dev_B, dev_len, &S);

    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(ans, dev_B.mem, sizeof(float4), cudaMemcpyDeviceToHost, S));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    float _ans = 0;
    _ans += ans->x;
    _ans += ans->y;
    _ans += ans->z;
    _ans += ans->w;
    *res = _ans;
    delete ans;

    decx::alloc::_device_dealloc(&dev_tmp);
    checkCudaErrors(cudaStreamDestroy(S));
}



void decx::Dot_fp32_ten(_Tensor<float>* A, _Tensor<float>* B, float* res, de::DH* handle)
{
    // 2x float4 as a pair of adding operation
    const size_t dev_len = decx::utils::ceil<size_t>(A->_element_num, 8) * 2;

    decx::PtrInfo<float4> dev_tmp;
    if (decx::alloc::_device_malloc(&dev_tmp, 2 * dev_len * sizeof(float4))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp.ptr;
    dev_B.mem = dev_tmp.ptr + dev_len;

    checkCudaErrors(cudaMemcpy2DAsync(dev_A.mem, A->width * A->dpitch * sizeof(float),
        A->Tens.ptr, A->dp_x_wp * sizeof(float),
        A->width * A->dpitch * sizeof(float), A->height, cudaMemcpyHostToDevice, S));

    checkCudaErrors(cudaMemcpy2DAsync(dev_B.mem, B->width * B->dpitch * sizeof(float),
        B->Tens.ptr, B->dp_x_wp * sizeof(float),
        B->width * B->dpitch * sizeof(float), B->height, cudaMemcpyHostToDevice, S));

    decx::utils::set_mutex_memory_state<float4, float4>(&dev_A, &dev_B);

    float4* ans = new float4();

    decx::Kdot_fp32(&dev_A, &dev_B, dev_len, &S);

    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(ans, dev_B.mem, sizeof(float4), cudaMemcpyDeviceToHost, S));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    float _ans = 0;
    _ans += ans->x;
    _ans += ans->y;
    _ans += ans->z;
    _ans += ans->w;
    *res = _ans;
    delete ans;

    decx::alloc::_device_dealloc(&dev_tmp);
    checkCudaErrors(cudaStreamDestroy(S));
}



de::DH de::cuda::Dot(de::Vector<float>& A, de::Vector<float>& B, float* res)
{
    decx::_Vector<float>* _A = dynamic_cast<decx::_Vector<float>*>(&A);
    decx::_Vector<float>* _B = dynamic_cast<decx::_Vector<float>*>(&B);

    de::DH handle;
    decx::Success(&handle);

    if (_A->length != _B->length) {
        decx::MDim_Not_Matching(&handle);
        Print_Error_Message(4, DIM_NOT_EQUAL);
        return handle;
    }

    decx::Dot_fp32_vec(_A, _B, res, &handle);
    return handle;
}


de::DH de::cuda::Dot(de::GPU_Vector<float>& A, de::GPU_Vector<float>& B, float* res)
{
    decx::_GPU_Vector<float>* _A = dynamic_cast<decx::_GPU_Vector<float>*>(&A);
    decx::_GPU_Vector<float>* _B = dynamic_cast<decx::_GPU_Vector<float>*>(&B);

    de::DH handle;
    decx::Success(&handle);

    if (_A->length != _B->length) {
        decx::MDim_Not_Matching(&handle);
        Print_Error_Message(4, DIM_NOT_EQUAL);
        return handle;
    }

    decx::dev_Dot_fp32_vec(_A, _B, res, &handle);
    return handle;
}


de::DH de::cuda::Dot(de::Matrix<float>& A, de::Matrix<float>& B, float* res)
{
    _Matrix<float>* _A = dynamic_cast<_Matrix<float>*>(&A);
    _Matrix<float>* _B = dynamic_cast<_Matrix<float>*>(&B);

    de::DH handle;
    decx::Success(&handle);

    if (_A->width != _B->width || _A->height != _B->height) {
        decx::MDim_Not_Matching(&handle);
        Print_Error_Message(4, DIM_NOT_EQUAL);
        return handle;
    }

    decx::Dot_fp32_mat(_A, _B, res, &handle);
    return handle;
}


de::DH de::cuda::Dot(de::Tensor<float>& A, de::Tensor<float>& B, float* res)
{
    _Tensor<float>* _A = dynamic_cast<_Tensor<float>*>(&A);
    _Tensor<float>* _B = dynamic_cast<_Tensor<float>*>(&B);

    de::DH handle;
    decx::Success(&handle);

    if (_A->width != _B->width || _A->height != _B->height || _A->depth != _B->depth) {
        decx::MDim_Not_Matching(&handle);
        Print_Error_Message(4, DIM_NOT_EQUAL);
        return handle;
    }

    decx::Dot_fp32_ten(_A, _B, res, &handle);
    return handle;
}


de::DH de::cuda::Dot(de::GPU_Matrix<float>& A, de::GPU_Matrix<float>& B, float* res)
{
    decx::_GPU_Matrix<float>* _A = dynamic_cast<decx::_GPU_Matrix<float>*>(&A);
    decx::_GPU_Matrix<float>* _B = dynamic_cast<decx::_GPU_Matrix<float>*>(&B);

    de::DH handle;
    decx::Success(&handle);

    if (_A->width != _B->width || _A->height != _B->height) {
        decx::MDim_Not_Matching(&handle);
        Print_Error_Message(4, DIM_NOT_EQUAL);
        return handle;
    }

    decx::dev_Dot_fp32_mat(_A, _B, res, &handle);
    return handle;
}




de::DH de::cuda::Dot(de::GPU_Tensor<float>& A, de::GPU_Tensor<float>& B, float* res)
{
    decx::_GPU_Tensor<float>* _A = dynamic_cast<decx::_GPU_Tensor<float>*>(&A);
    decx::_GPU_Tensor<float>* _B = dynamic_cast<decx::_GPU_Tensor<float>*>(&B);

    de::DH handle;
    decx::Success(&handle);

    if (_A->width != _B->width || _A->height != _B->height || _A->depth != _B->depth) {
        decx::MDim_Not_Matching(&handle);
        Print_Error_Message(4, DIM_NOT_EQUAL);
        return handle;
    }

    decx::dev_Dot_fp32_ten(_A, _B, res, &handle);
    return handle;
}



#endif