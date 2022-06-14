/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _CUDA_GEMM_H_
#define _CUDA_GEMM_H_

#include "GEMM.cuh"
#include "GEMM_nonuniform.cuh"
#include "../../../classes/Matrix.h"
#include "../../../classes/GPU_Matrix.h"
#include "../../../classes/classes_util.h"
#include "GEMM3_macros.h"
#include "../../../core/memory_management/Memory_pool.h"



namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH GEMM(de::Matrix<float>& A, de::Matrix<float>& B, de::Matrix<float>& dst);


        _DECX_API_ de::DH GEMM(de::Matrix<float>& A, de::Matrix<float>& B, de::Matrix<float>& C, de::Matrix<float>& dst);


        _DECX_API_ de::DH GEMM(de::Matrix<de::Half>& A, de::Matrix<de::Half>& B, de::Matrix<de::Half>& dst);


        _DECX_API_ de::DH GEMM(de::Matrix<de::Half>& A, de::Matrix<de::Half>& B, de::Matrix<de::Half>& C, de::Matrix<de::Half>& dst);

        // --------------------------------------------- pure GPU -----------------------------------------------------------

        _DECX_API_ de::DH GEMM(de::GPU_Matrix<float>& A, de::GPU_Matrix<float>& B, de::GPU_Matrix<float>& dst);


        _DECX_API_ de::DH GEMM(de::GPU_Matrix<de::Half>& A, de::GPU_Matrix<de::Half>& B, de::GPU_Matrix<de::Half>& dst);


        _DECX_API_ de::DH GEMM(de::GPU_Matrix<float>& A, de::GPU_Matrix<float>& B, de::GPU_Matrix<float>& C, de::GPU_Matrix<float>& dst);


        _DECX_API_ de::DH GEMM(de::GPU_Matrix<de::Half>& A, de::GPU_Matrix<de::Half>& B, de::GPU_Matrix<de::Half>& C, de::GPU_Matrix<de::Half>& dst);
    }
}




de::DH de::cuda::GEMM(de::Matrix<float>& A, de::Matrix<float>& B, de::Matrix<float>& dst)
{
    decx::_Matrix<float>& _A = dynamic_cast<decx::_Matrix<float>&>(A);
    decx::_Matrix<float>& _B = dynamic_cast<decx::_Matrix<float>&>(B);
    decx::_Matrix<float>& _dst = dynamic_cast<decx::_Matrix<float>&>(dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_A.width != _B.height) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    _dst.re_construct(_B.width, _A.height, decx::DATA_STORE_TYPE::Page_Locked);

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A.width % 16) != 0)    ? (_cu_ceil(_A.width, 16) * 16)        : _A.width;
    pitch_B = ((_B.width % 128) != 0)    ? (_cu_ceil(_B.width, 128) * 128)    : _B.width;
    hA        = ((_A.height % 128) != 0)    ? (_cu_ceil(_A.height, 128) * 128)    : _A.height;
    
    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    decx::PtrInfo<float> DA, DB, Ddst;

    if (decx::alloc::_device_malloc(&DA, mem_A * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(&handle);
        return handle;
    }
    if (decx::alloc::_device_malloc(&DB, mem_B * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(&handle);
        return handle;
    }
    if (decx::alloc::_device_malloc(&Ddst, mem_dst * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(&handle);
        return handle;
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));
    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DA.ptr, pitch_A * sizeof(float), _A.Mat.ptr, _A.pitch * sizeof(float),
        _A.width * sizeof(float), _A.height, cudaMemcpyHostToDevice, S));

    // copy B from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DB.ptr, pitch_B * sizeof(float), _B.Mat.ptr, _B.pitch * sizeof(float),
        _B.width * sizeof(float), _B.height, cudaMemcpyHostToDevice, S));

    decx::sGEMM_part(DA.ptr, DB.ptr, Ddst.ptr, pitch_A, pitch_B, hA, &S);

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(_dst.Mat.ptr, _dst.pitch * sizeof(float), Ddst.ptr, pitch_B * sizeof(float),
        _B.width * sizeof(float), _A.height, cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    decx::alloc::_device_dealloc(&DA);
    decx::alloc::_device_dealloc(&DB);
    decx::alloc::_device_dealloc(&Ddst);
    
    decx::Success(&handle);
    return handle;
}




de::DH de::cuda::GEMM(de::Matrix<float>& A, de::Matrix<float>& B, de::Matrix<float>& C, de::Matrix<float>& dst)
{
    decx::_Matrix<float>& _A = dynamic_cast<decx::_Matrix<float>&>(A);
    decx::_Matrix<float>& _B = dynamic_cast<decx::_Matrix<float>&>(B);
    decx::_Matrix<float>& _C = dynamic_cast<decx::_Matrix<float>&>(C);
    decx::_Matrix<float>& _dst = dynamic_cast<decx::_Matrix<float>&>(dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_A.width != _B.height) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    if (_A.height != _C.height || _B.width != _C.width) {
        decx::GEMM_DimNMatch(&handle);
        return handle;
    }

    _dst.re_construct(_B.width, _A.height, decx::DATA_STORE_TYPE::Page_Locked);

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A.width % 16) != 0)    ? (_cu_ceil(_A.width, 16) * 16)        : _A.width;
    pitch_B = ((_B.width % 128) != 0)    ? (_cu_ceil(_B.width, 128) * 128)    : _B.width;
    hA        = ((_A.height % 128) != 0)    ? (_cu_ceil(_A.height, 128) * 128)    : _A.height;

    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    decx::PtrInfo<float> DA, DB, Ddst;
    if (decx::alloc::_device_malloc(&DA, mem_A * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(&handle);
        return handle;
    }
    if (decx::alloc::_device_malloc(&DB, mem_B * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(&handle);
        return handle;
    }
    if (decx::alloc::_device_malloc(&Ddst, mem_dst * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(&handle);
        return handle;
    }
    
    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DA.ptr, pitch_A * sizeof(float), _A.Mat.ptr, _A.pitch * sizeof(float),
        _A.width * sizeof(float), _A.height, cudaMemcpyHostToDevice, S));

    // copy B from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DB.ptr, pitch_B * sizeof(float), _B.Mat.ptr, _B.pitch * sizeof(float),
        _B.width * sizeof(float), _B.height, cudaMemcpyHostToDevice, S));

    // contemparay storage to dst, to save device memory
    checkCudaErrors(cudaMemcpy2DAsync(Ddst.ptr, pitch_B * sizeof(float), _C.Mat.ptr, _C.pitch * sizeof(float),
        _C.width * sizeof(float), _C.height, cudaMemcpyHostToDevice, S));

    decx::sGEMM_part_ABC(DA.ptr, DB.ptr, Ddst.ptr, Ddst.ptr, pitch_A, pitch_B, hA, &S);

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(_dst.Mat.ptr, _dst.pitch * sizeof(float), Ddst.ptr, pitch_B * sizeof(float),
        _B.width * sizeof(float), _A.height, cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));
    
    decx::alloc::_device_dealloc(&DA);
    decx::alloc::_device_dealloc(&DB);
    decx::alloc::_device_dealloc(&Ddst);

    decx::Success(&handle);
    return handle;
}





de::DH de::cuda::GEMM(de::Matrix<de::Half>& A, de::Matrix<de::Half>& B, de::Matrix<de::Half>& dst)
{
    decx::_Matrix<de::Half>& _A = dynamic_cast<decx::_Matrix<de::Half>&>(A);
    decx::_Matrix<de::Half>& _B = dynamic_cast<decx::_Matrix<de::Half>&>(B);
    decx::_Matrix<de::Half>& _dst = dynamic_cast<decx::_Matrix<de::Half>&>(dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_A.width != _B.height) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    _dst.re_construct(_B.width, _A.height, decx::DATA_STORE_TYPE::Page_Locked);

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A.width % 16) != 0)    ? (_cu_ceil(_A.width, 16) * 16)        : _A.width;
    pitch_B = ((_B.width % 128) != 0)    ? (_cu_ceil(_B.width, 128) * 128)    : _B.width;
    hA        = ((_A.height % 128) != 0)    ? (_cu_ceil(_A.height, 128) * 128)    : _A.height;
    
    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    decx::PtrInfo<de::Half> DA, DB, Ddst;
    if (decx::alloc::_device_malloc(&DA, mem_A * sizeof(de::Half))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(&handle);
        return handle;
    }
    if (decx::alloc::_device_malloc(&DB, mem_B * sizeof(de::Half))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(&handle);
        return handle;
    }
    if (decx::alloc::_device_malloc(&Ddst, mem_dst * sizeof(de::Half))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(&handle);
        return handle;
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));
    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DA.ptr, pitch_A * sizeof(de::Half), _A.Mat.ptr, _A.pitch * sizeof(de::Half),
        _A.width * sizeof(de::Half), _A.height, cudaMemcpyHostToDevice, S));
    
    // copy B from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DB.ptr, pitch_B * sizeof(de::Half), _B.Mat.ptr, _B.pitch * sizeof(de::Half),
        _B.width * sizeof(de::Half), _B.height, cudaMemcpyHostToDevice, S));

    decx::hGEMM_part(DA.ptr, DB.ptr, Ddst.ptr, pitch_A, pitch_B, hA, &S);
    
    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(_dst.Mat.ptr, _dst.pitch * sizeof(de::Half), Ddst.ptr, pitch_B * sizeof(de::Half),
        _B.width * sizeof(de::Half), _A.height, cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));
    
    decx::alloc::_device_dealloc(&DA);
    decx::alloc::_device_dealloc(&DB);
    decx::alloc::_device_dealloc(&Ddst);

    decx::Success(&handle);
    return handle;
}



de::DH de::cuda::GEMM(de::Matrix<de::Half>& A, de::Matrix<de::Half>& B, de::Matrix<de::Half>& C, de::Matrix<de::Half>& dst)
{
    decx::_Matrix<de::Half>& _A = dynamic_cast<decx::_Matrix<de::Half>&>(A);
    decx::_Matrix<de::Half>& _B = dynamic_cast<decx::_Matrix<de::Half>&>(B);
    decx::_Matrix<de::Half>& _C = dynamic_cast<decx::_Matrix<de::Half>&>(C);
    decx::_Matrix<de::Half>& _dst = dynamic_cast<decx::_Matrix<de::Half>&>(dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_A.width != _B.height) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    if (_A.height != _C.height || _B.width != _C.width) {
        decx::GEMM_DimNMatch(&handle);
        return handle;
    }

    _dst.re_construct(_B.width, _A.height, decx::DATA_STORE_TYPE::Page_Locked);

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A.width % 16) != 0) ? (_cu_ceil(_A.width, 16) * 16) : _A.width;
    pitch_B = ((_B.width % 128) != 0) ? (_cu_ceil(_B.width, 128) * 128) : _B.width;
    hA = ((_A.height % 128) != 0) ? (_cu_ceil(_A.height, 128) * 128) : _A.height;

    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    decx::PtrInfo<de::Half> DA, DB, Ddst;
    if (decx::alloc::_device_malloc(&DA, mem_A * sizeof(de::Half))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(&handle);
        return handle;
    }
    if (decx::alloc::_device_malloc(&DB, mem_B * sizeof(de::Half))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(&handle);
        return handle;
    }
    if (decx::alloc::_device_malloc(&Ddst, mem_dst * sizeof(de::Half))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(&handle);
        return handle;
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DA.ptr, pitch_A * sizeof(de::Half), _A.Mat.ptr, _A.pitch * sizeof(de::Half),
        _A.width * sizeof(de::Half), _A.height, cudaMemcpyHostToDevice, S));

    // copy B from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DB.ptr, pitch_B * sizeof(de::Half), _B.Mat.ptr, _B.pitch * sizeof(de::Half),
        _B.width * sizeof(de::Half), _B.height, cudaMemcpyHostToDevice, S));

    // contemparay storage to dst, to save device memory
    checkCudaErrors(cudaMemcpy2DAsync(Ddst.ptr, pitch_B * sizeof(de::Half), _C.Mat.ptr, _C.pitch * sizeof(de::Half),
        _C.width * sizeof(de::Half), _C.height, cudaMemcpyHostToDevice, S));

    decx::hGEMM_part_ABC(DA.ptr, DB.ptr, Ddst.ptr, Ddst.ptr, pitch_A, pitch_B, hA, &S);

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(_dst.Mat.ptr, _dst.pitch * sizeof(de::Half), Ddst.ptr, pitch_B * sizeof(de::Half),
        _B.width * sizeof(de::Half), _A.height, cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    
    decx::alloc::_device_dealloc(&DA);
    decx::alloc::_device_dealloc(&DB);
    decx::alloc::_device_dealloc(&Ddst);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    decx::Success(&handle);
    return handle;
}


// --------------------------------------------- pure GPU ------------------------------------------------


de::DH de::cuda::GEMM(de::GPU_Matrix<float>& A, de::GPU_Matrix<float>& B, de::GPU_Matrix<float>& dst)
{
    decx::_GPU_Matrix<float>& _A = dynamic_cast<decx::_GPU_Matrix<float>&>(A);
    decx::_GPU_Matrix<float>& _B = dynamic_cast<decx::_GPU_Matrix<float>&>(B);
    decx::_GPU_Matrix<float>& _dst = dynamic_cast<decx::_GPU_Matrix<float>&>(dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_A.width != _B.height) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }
    _dst.re_construct(_B.width, _A.height);

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    const uint block = 256;
    const dim3 grid(decx::utils::ceil<uint>(_A.height, 16 * 8), decx::utils::ceil<uint>(_B.pitch, 16 * 8));

    if ((_B.pitch % 128) || (_A.height % 128)) {        // dstdims CAN NOT be divided into integers
        if (_B.height % 16) {
            cu_GEMM_fp32_anyWH_anyL << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(_A.Mat.ptr), reinterpret_cast<float4*>(_B.Mat.ptr),
                reinterpret_cast<float4*>(_dst.Mat.ptr), _A.pitch / 4, _B.pitch / 4, _A.height, _B.height,
                decx::utils::ceil<uint>(_A.pitch, 16));
        }
        else {
            cu_GEMM_fp32_anyWH_specL << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(_A.Mat.ptr), reinterpret_cast<float4*>(_B.Mat.ptr),
                reinterpret_cast<float4*>(_dst.Mat.ptr), _A.pitch / 4, _B.pitch / 4, _A.height, _A.pitch / 16);
        }
    }
    else {
        if (_B.height % 16) {
            cu_GEMM_fp32_specWH_anyL << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(_A.Mat.ptr), reinterpret_cast<float4*>(_B.Mat.ptr),
                reinterpret_cast<float4*>(_dst.Mat.ptr), _A.pitch / 4, _B.pitch / 4, _B.height,
                decx::utils::ceil<uint>(_A.pitch, 16));
        }
        else {
            cu_GEMM_fp32_spec << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(_A.Mat.ptr), reinterpret_cast<float4*>(_B.Mat.ptr),
                reinterpret_cast<float4*>(_dst.Mat.ptr), _A.pitch / 4, _B.pitch / 4, _B.height / 16);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    decx::Success(&handle);
    return handle;
}


de::DH de::cuda::GEMM(de::GPU_Matrix<de::Half>& A, de::GPU_Matrix<de::Half>& B, de::GPU_Matrix<de::Half>& dst)
{
    decx::_GPU_Matrix<de::Half>& _A = dynamic_cast<decx::_GPU_Matrix<de::Half>&>(A);
    decx::_GPU_Matrix<de::Half>& _B = dynamic_cast<decx::_GPU_Matrix<de::Half>&>(B);
    decx::_GPU_Matrix<de::Half>& _dst = dynamic_cast<decx::_GPU_Matrix<de::Half>&>(dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_A.width != _B.height) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    if (_A.height != _dst.height || _B.width != _dst.width) {
        decx::GEMM_DimNMatch(&handle);
        return handle;
    }

    _dst.re_construct(_B.width, _A.height);

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    const uint block = 256;
    const dim3 grid(decx::utils::ceil<uint>(_A.height, 16 * 8), decx::utils::ceil<uint>(_B.pitch, 16 * 8));

    if (_B.pitch % 128 || _A.height % 128) {
        if (_B.height % 16) {
            cu_GEMM_fp16_anyWH_anyL << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(_A.Mat.ptr), reinterpret_cast<float4*>(_B.Mat.ptr),
                reinterpret_cast<float4*>(_dst.Mat.ptr), _A.pitch / 8, _B.pitch / 8, _A.height, _B.height,
                decx::utils::ceil<uint>(_A.pitch, 16));
        }
        else {
            cu_GEMM_fp16_anyWH_specL << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(_A.Mat.ptr), reinterpret_cast<float4*>(_B.Mat.ptr),
                reinterpret_cast<float4*>(_dst.Mat.ptr), _A.pitch / 8, _B.pitch / 8, _A.height, _A.pitch / 16);
        }
    }
    else {
        if (_B.height % 16) {
            cu_GEMM_fp16_specWH_anyL << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(_A.Mat.ptr), reinterpret_cast<float4*>(_B.Mat.ptr),
                reinterpret_cast<float4*>(_dst.Mat.ptr), _A.pitch / 8, _B.pitch / 8, _B.height,
                decx::utils::ceil<uint>(_A.pitch, 16));
        }
        else {
            cu_GEMM_fp16_spec << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(_A.Mat.ptr), reinterpret_cast<float4*>(_B.Mat.ptr),
                reinterpret_cast<float4*>(_dst.Mat.ptr), _A.pitch / 8, _B.pitch / 8, _A.pitch / 16);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    decx::Success(&handle);
    return handle;
}


de::DH de::cuda::GEMM(de::GPU_Matrix<float>& A, de::GPU_Matrix<float>& B, de::GPU_Matrix<float>& C, de::GPU_Matrix<float>& dst)
{
    decx::_GPU_Matrix<float>& _A = dynamic_cast<decx::_GPU_Matrix<float>&>(A);
    decx::_GPU_Matrix<float>& _B = dynamic_cast<decx::_GPU_Matrix<float>&>(B);
    decx::_GPU_Matrix<float>& _C = dynamic_cast<decx::_GPU_Matrix<float>&>(C);
    decx::_GPU_Matrix<float>& _dst = dynamic_cast<decx::_GPU_Matrix<float>&>(dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_A.width != _B.height) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    _dst.re_construct(_B.width, _A.height);

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    const uint block = 256;
    const dim3 grid(decx::utils::ceil<uint>(_A.height, 16 * 8), decx::utils::ceil<uint>(_B.pitch, 16 * 8));

    if ((_B.pitch % 128) || (_A.height % 128)) {        // dstdims CAN NOT be divided into integers
        if (_B.height % 16) {
            cu_GEMM_fp32_ABC_anyWH_anyL << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(_A.Mat.ptr), reinterpret_cast<float4*>(_B.Mat.ptr),
                reinterpret_cast<float4*>(_C.Mat.ptr), reinterpret_cast<float4*>(_dst.Mat.ptr), 
                _A.pitch / 4, _B.pitch / 4, _A.height, _B.height, decx::utils::ceil<uint>(_A.pitch, 16));
        }
        else {
            cu_GEMM_fp32_ABC_anyWH_specL << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(_A.Mat.ptr), reinterpret_cast<float4*>(_B.Mat.ptr),
                reinterpret_cast<float4*>(_C.Mat.ptr), reinterpret_cast<float4*>(_dst.Mat.ptr), 
                _A.pitch / 4, _B.pitch / 4, _A.height, _A.pitch / 16);
        }
    }
    else {
        if (_B.height % 16) {
            cu_GEMM_fp32_ABC_specWH_anyL << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(_A.Mat.ptr), reinterpret_cast<float4*>(_B.Mat.ptr),
                reinterpret_cast<float4*>(_C.Mat.ptr), reinterpret_cast<float4*>(_dst.Mat.ptr),
                _A.pitch / 4, _B.pitch / 4, _B.height,  decx::utils::ceil<uint>(_A.pitch, 16));
        }
        else {
            cu_GEMM_fp32_ABC_spec << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(_A.Mat.ptr), reinterpret_cast<float4*>(_B.Mat.ptr),
                reinterpret_cast<float4*>(_C.Mat.ptr), reinterpret_cast<float4*>(_dst.Mat.ptr),
                _A.pitch / 4, _B.pitch / 4, _B.height / 16);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    decx::Success(&handle);
    return handle;
}


de::DH de::cuda::GEMM(de::GPU_Matrix<de::Half>& A, de::GPU_Matrix<de::Half>& B, de::GPU_Matrix<de::Half>& C, de::GPU_Matrix<de::Half>& dst)
{
    decx::_GPU_Matrix<de::Half>& _A = dynamic_cast<decx::_GPU_Matrix<de::Half>&>(A);
    decx::_GPU_Matrix<de::Half>& _B = dynamic_cast<decx::_GPU_Matrix<de::Half>&>(B);
    decx::_GPU_Matrix<de::Half>& _C = dynamic_cast<decx::_GPU_Matrix<de::Half>&>(C);
    decx::_GPU_Matrix<de::Half>& _dst = dynamic_cast<decx::_GPU_Matrix<de::Half>&>(dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_A.width != _B.height) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    _dst.re_construct(_B.width, _A.height);

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    const uint block = 256;
    const dim3 grid(decx::utils::ceil<uint>(_A.height, 16 * 8), decx::utils::ceil<uint>(_B.pitch, 16 * 8));

    if ((_B.pitch % 128) || (_A.height % 128)) {        // dstdims CAN NOT be divided into integers
        if (_B.height % 16) {
            cu_GEMM_fp16_ABC_anyWH_anyL << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(_A.Mat.ptr), reinterpret_cast<float4*>(_B.Mat.ptr),
                reinterpret_cast<float4*>(_C.Mat.ptr), reinterpret_cast<float4*>(_dst.Mat.ptr),
                _A.pitch / 8, _B.pitch / 8, _A.height, _B.height, decx::utils::ceil<uint>(_A.pitch, 16));
        }
        else {
            cu_GEMM_fp16_ABC_anyWH_specL << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(_A.Mat.ptr), reinterpret_cast<float4*>(_B.Mat.ptr),
                reinterpret_cast<float4*>(_C.Mat.ptr), reinterpret_cast<float4*>(_dst.Mat.ptr),
                _A.pitch / 8, _B.pitch / 8, _A.height, _A.pitch / 16);
        }
    }
    else {
        if (_B.height % 16) {
            cu_GEMM_fp16_ABC_specWH_anyL << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(_A.Mat.ptr), reinterpret_cast<float4*>(_B.Mat.ptr),
                reinterpret_cast<float4*>(_C.Mat.ptr), reinterpret_cast<float4*>(_dst.Mat.ptr),
                _A.pitch / 8, _B.pitch / 8, _B.height, decx::utils::ceil<uint>(_A.pitch, 16));
        }
        else {
            cu_GEMM_fp16_ABC_spec << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(_A.Mat.ptr), reinterpret_cast<float4*>(_B.Mat.ptr),
                reinterpret_cast<float4*>(_C.Mat.ptr), reinterpret_cast<float4*>(_dst.Mat.ptr),
                _A.pitch / 8, _B.pitch / 8, _B.height / 16);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    decx::Success(&handle);
    return handle;
}


#endif
