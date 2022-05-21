/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#include "GEMM.cuh"
#include "../../../classes/Matrix.h"
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
    }
}





de::DH de::cuda::GEMM(_FLOAT_& A, _FLOAT_& B, _FLOAT_& dst)
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

    if (_A.height != _dst.height || _B.width != _dst.width) {
        decx::GEMM_DimNMatch(&handle);
        return handle;
    }

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A.width % 16) != 0)    ? (_cu_ceil(_A.width, 16) * 16)        : _A.width;
    pitch_B = ((_B.width % 128) != 0)    ? (_cu_ceil(_B.width, 128) * 128)    : _B.width;
    hA        = ((_A.height % 128) != 0)    ? (_cu_ceil(_A.height, 128) * 128)    : _A.height;
    
    float* DA, * DB, * Ddst;
    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    decx::PtrInfo<float> dev_tmp;
    decx::alloc::_alloc_D(&dev_tmp.block, (mem_A + mem_B + mem_dst) * sizeof(float));
    dev_tmp._sync_type();

    DA = dev_tmp.ptr;
    DB = DA + mem_A;
    Ddst = DB + mem_B;

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));
    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DA, pitch_A * sizeof(float), _A.Mat.ptr, _A.pitch * sizeof(float),
        _A.width * sizeof(float), _A.height, cudaMemcpyHostToDevice, S));

    // copy B from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DB, pitch_B * sizeof(float), _B.Mat.ptr, _B.pitch * sizeof(float),
        _B.width * sizeof(float), _B.height, cudaMemcpyHostToDevice, S));

    decx::sGEMM_part(DA, DB, Ddst, pitch_A, pitch_B, hA, &S);

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(_dst.Mat.ptr, _dst.pitch * sizeof(float), Ddst, pitch_B * sizeof(float),
        _B.width * sizeof(float), _A.height, cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    decx::alloc::_dealloc_D(dev_tmp.block);
    
    decx::Success(&handle);
    return handle;
}




de::DH de::cuda::GEMM(_FLOAT_& A, _FLOAT_& B, _FLOAT_& C, _FLOAT_& dst)
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

    if (_A.height != _dst.height || _B.width != _dst.width) {
        decx::GEMM_DimNMatch(&handle);
        return handle;
    }

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A.width % 16) != 0)    ? (_cu_ceil(_A.width, 16) * 16)        : _A.width;
    pitch_B = ((_B.width % 128) != 0)    ? (_cu_ceil(_B.width, 128) * 128)    : _B.width;
    hA        = ((_A.height % 128) != 0)    ? (_cu_ceil(_A.height, 128) * 128)    : _A.height;

    float* DA, * DB, * Ddst;
    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    decx::PtrInfo<float> dev_tmp;
    decx::alloc::_alloc_D(&dev_tmp.block, (mem_A + mem_B + mem_dst) * sizeof(float));
    dev_tmp._sync_type();

    DA = dev_tmp.ptr;
    DB = DA + mem_A;
    Ddst = DB + mem_B;

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DA, pitch_A * sizeof(float), _A.Mat.ptr, _A.pitch * sizeof(float),
        _A.width * sizeof(float), _A.height, cudaMemcpyHostToDevice, S));

    // copy B from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DB, pitch_B * sizeof(float), _B.Mat.ptr, _B.pitch * sizeof(float),
        _B.width * sizeof(float), _B.height, cudaMemcpyHostToDevice, S));

    // contemparay storage to dst, to save device memory
    checkCudaErrors(cudaMemcpy2DAsync(Ddst, pitch_B * sizeof(float), _C.Mat.ptr, _C.pitch * sizeof(float),
        _C.width * sizeof(float), _C.height, cudaMemcpyHostToDevice, S));

    decx::sGEMM_part_ABC(DA, DB, Ddst, Ddst, pitch_A, pitch_B, hA, &S);

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(_dst.Mat.ptr, _dst.pitch * sizeof(float), Ddst, pitch_B * sizeof(float),
        _B.width * sizeof(float), _A.height, cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));
    
    decx::alloc::_dealloc_D(dev_tmp.block);

    decx::Success(&handle);
    return handle;
}





de::DH de::cuda::GEMM(_HALF_& A, _HALF_& B, _HALF_& dst)
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

    if (_A.height != _dst.height || _B.width != _dst.width) {
        decx::GEMM_DimNMatch(&handle);
        return handle;
    }

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A.width % 16) != 0)    ? (_cu_ceil(_A.width, 16) * 16)        : _A.width;
    pitch_B = ((_B.width % 128) != 0)    ? (_cu_ceil(_B.width, 128) * 128)    : _B.width;
    hA        = ((_A.height % 128) != 0)    ? (_cu_ceil(_A.height, 128) * 128)    : _A.height;
    
    de::Half* DA, * DB, * Ddst;
    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    decx::PtrInfo<de::Half> dev_tmp;
    decx::alloc::_alloc_D(&dev_tmp.block, (mem_A + mem_B + mem_dst) * sizeof(de::Half));
    dev_tmp._sync_type();

    DA = dev_tmp.ptr;
    DB = DA + mem_A;
    Ddst = DB + mem_B;

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));
    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DA, pitch_A * sizeof(de::Half), _A.Mat.ptr, _A.pitch * sizeof(de::Half),
        _A.width * sizeof(de::Half), _A.height, cudaMemcpyHostToDevice, S));
    
    // copy B from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DB, pitch_B * sizeof(de::Half), _B.Mat.ptr, _B.pitch * sizeof(de::Half),
        _B.width * sizeof(de::Half), _B.height, cudaMemcpyHostToDevice, S));

    decx::hGEMM_part(DA, DB, Ddst, pitch_A, pitch_B, hA, &S);
    
    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(_dst.Mat.ptr, _dst.pitch * sizeof(de::Half), Ddst, pitch_B * sizeof(de::Half),
        _B.width * sizeof(de::Half), _A.height, cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));
    
    decx::alloc::_dealloc_D(dev_tmp.block);

    decx::Success(&handle);
    return handle;
}



de::DH de::cuda::GEMM(_HALF_& A, _HALF_& B, _HALF_& C, _HALF_& dst)
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

    if (_A.height != _dst.height || _B.width != _dst.width) {
        decx::GEMM_DimNMatch(&handle);
        return handle;
    }

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A.width % 16) != 0) ? (_cu_ceil(_A.width, 16) * 16) : _A.width;
    pitch_B = ((_B.width % 128) != 0) ? (_cu_ceil(_B.width, 128) * 128) : _B.width;
    hA = ((_A.height % 128) != 0) ? (_cu_ceil(_A.height, 128) * 128) : _A.height;

    de::Half* DA, * DB, * Ddst;
    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    decx::PtrInfo<de::Half> dev_tmp;
    decx::alloc::_alloc_D(&dev_tmp.block, (mem_A + mem_B + mem_dst) * sizeof(de::Half));
    dev_tmp._sync_type();

    DA = dev_tmp.ptr;
    DB = DA + mem_A;
    Ddst = DB + mem_B;

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DA, pitch_A * sizeof(de::Half), _A.Mat.ptr, _A.pitch * sizeof(de::Half),
        _A.width * sizeof(de::Half), _A.height, cudaMemcpyHostToDevice, S));

    // copy B from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DB, pitch_B * sizeof(de::Half), _B.Mat.ptr, _B.pitch * sizeof(de::Half),
        _B.width * sizeof(de::Half), _B.height, cudaMemcpyHostToDevice, S));

    // contemparay storage to dst, to save device memory
    checkCudaErrors(cudaMemcpy2DAsync(Ddst, pitch_B * sizeof(de::Half), _C.Mat.ptr, _C.pitch * sizeof(de::Half),
        _C.width * sizeof(de::Half), _C.height, cudaMemcpyHostToDevice, S));

    decx::hGEMM_part_ABC(DA, DB, Ddst, Ddst, pitch_A, pitch_B, hA, &S);

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(_dst.Mat.ptr, _dst.pitch * sizeof(de::Half), Ddst, pitch_B * sizeof(de::Half),
        _B.width * sizeof(de::Half), _A.height, cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    
    decx::alloc::_dealloc_D(dev_tmp.block);
    checkCudaErrors(cudaStreamDestroy(S));

    decx::Success(&handle);
    return handle;
}