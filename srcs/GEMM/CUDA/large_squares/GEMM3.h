/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#include "../../../classes/core_types.h"
#include "../../../core/defines.h"
#include "GEMM3_macros.h"
#include "../../../core/memory_management/Memory_pool.h"


using de::MatrixArray;
using decx::_MatrixArray;


namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH GEMM3(MatrixArray<float>& A, MatrixArray<float>& B, MatrixArray<float>& dst);


        _DECX_API_ de::DH GEMM3(MatrixArray<de::Half>& A, MatrixArray<de::Half>& B, MatrixArray<de::Half>& dst);


        _DECX_API_ de::DH GEMM3(MatrixArray<float>& A, MatrixArray<float>& B, MatrixArray<float>& C, MatrixArray<float>& dst);


        _DECX_API_ de::DH GEMM3(MatrixArray<de::Half>& A, MatrixArray<de::Half>& B, MatrixArray<de::Half>& C, MatrixArray<de::Half>& dst);
    }
}






de::DH de::cuda::GEMM3(MatrixArray<float>& A, MatrixArray<float>& B, MatrixArray<float>& dst)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    _MatrixArray<float>* _A = dynamic_cast<_MatrixArray<float>*>(&A);
    _MatrixArray<float>* _B = dynamic_cast<_MatrixArray<float>*>(&B);
    _MatrixArray<float>* _dst = dynamic_cast<_MatrixArray<float>*>(&dst);

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A->width % 16) != 0)    ? (_cu_ceil(_A->width, 16) * 16)    : _A->width;
    pitch_B = ((_B->width % 128) != 0)    ? (_cu_ceil(_B->width, 128) * 128)    : _B->width;
    hA        = ((_A->height % 128) != 0) ? (_cu_ceil(_A->height, 128) * 128) : _A->height;

    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    /*
    * [0, 3] -> A;
    * [1, 4] -> B;
    * [2, 5] -> dst;
    */
    decx::PtrInfo<float> dev_tmp;            // this is the total buffer that this function requires
    decx::alloc::_alloc_D(&dev_tmp.block, (mem_A + mem_B + mem_dst) * 2 * sizeof(float));
    dev_tmp._sync_type();

    decx::alloc::MIF<float> d_mem[6];
    d_mem[0].mem = dev_tmp.ptr;
    //checkCudaErrors(cudaMalloc(&(d_mem[0].mem), (mem_A + mem_B + mem_dst) * 2 * sizeof(float)));
    d_mem[1].mem = d_mem[0].mem + mem_A;
    d_mem[2].mem = d_mem[1].mem + mem_B;
    d_mem[3].mem = d_mem[2].mem + mem_dst;
    d_mem[4].mem = d_mem[3].mem + mem_A;
    d_mem[5].mem = d_mem[4].mem + mem_B;

    cudaStream_t S[3];
    checkCudaErrors(cudaStreamCreate(&S[0]));
    checkCudaErrors(cudaStreamCreate(&S[1]));
    checkCudaErrors(cudaStreamCreate(&S[2]));

    GEMM3_part(float, decx::sGEMM_part)

    checkCudaErrors(cudaDeviceSynchronize());

    //checkCudaErrors(cudaFree(d_mem[0].mem));

    decx::alloc::_dealloc_D(dev_tmp.block);

    checkCudaErrors(cudaStreamDestroy(S[0]));
    checkCudaErrors(cudaStreamDestroy(S[1]));
    checkCudaErrors(cudaStreamDestroy(S[2]));

    decx::Success(&handle);
    return handle;
}




de::DH de::cuda::GEMM3(MatrixArray<de::Half>& A, MatrixArray<de::Half>& B, MatrixArray<de::Half>& dst)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    _MatrixArray<de::Half>* _A = dynamic_cast<_MatrixArray<de::Half>*>(&A);
    _MatrixArray<de::Half>* _B = dynamic_cast<_MatrixArray<de::Half>*>(&B);
    _MatrixArray<de::Half>* _dst = dynamic_cast<_MatrixArray<de::Half>*>(&dst);

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A->width % 16) != 0)    ? (_cu_ceil(_A->width, 16) * 16)    : _A->width;
    pitch_B = ((_B->width % 128) != 0)    ? (_cu_ceil(_B->width, 128) * 128)    : _B->width;
    hA        = ((_A->height % 128) != 0) ? (_cu_ceil(_A->height, 128) * 128) : _A->height;

    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    /*
    * [0, 3] -> A;
    * [1, 4] -> B;
    * [2, 5] -> dst;
    */
    decx::alloc::MIF<de::Half> d_mem[6];
    checkCudaErrors(cudaMalloc(&(d_mem[0].mem), (mem_A + mem_B + mem_dst) * 2 * sizeof(de::Half)));
    d_mem[1].mem = d_mem[0].mem + mem_A;
    d_mem[2].mem = d_mem[1].mem + mem_B;
    d_mem[3].mem = d_mem[2].mem + mem_dst;
    d_mem[4].mem = d_mem[3].mem + mem_A;
    d_mem[5].mem = d_mem[4].mem + mem_B;

    cudaStream_t S[3];
    checkCudaErrors(cudaStreamCreate(&S[0]));
    checkCudaErrors(cudaStreamCreate(&S[1]));
    checkCudaErrors(cudaStreamCreate(&S[2]));

    GEMM3_part(de::Half, decx::hGEMM_part)

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(d_mem[0].mem));
    checkCudaErrors(cudaStreamDestroy(S[0]));
    checkCudaErrors(cudaStreamDestroy(S[1]));
    checkCudaErrors(cudaStreamDestroy(S[2]));

    decx::Success(&handle);
    return handle;
}





de::DH de::cuda::GEMM3(MatrixArray<float>& A, MatrixArray<float>& B, MatrixArray<float>& C, MatrixArray<float>& dst)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    _MatrixArray<float>* _A = dynamic_cast<_MatrixArray<float>*>(&A);
    _MatrixArray<float>* _B = dynamic_cast<_MatrixArray<float>*>(&B);
    _MatrixArray<float>* _C = dynamic_cast<_MatrixArray<float>*>(&C);
    _MatrixArray<float>* _dst = dynamic_cast<_MatrixArray<float>*>(&dst);

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A->width % 16) != 0)    ? (_cu_ceil(_A->width, 16) * 16)    : _A->width;
    pitch_B = ((_B->width % 128) != 0)    ? (_cu_ceil(_B->width, 128) * 128)    : _B->width;
    hA        = ((_A->height % 128) != 0) ? (_cu_ceil(_A->height, 128) * 128) : _A->height;

    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    /*
    * [0, 3] -> A;
    * [1, 4] -> B;
    * [2, 5] -> dst;
    */
    decx::alloc::MIF<float> d_mem[6];
    checkCudaErrors(cudaMalloc(&(d_mem[0].mem), (mem_A + mem_B + mem_dst) * 2 * sizeof(float)));
    d_mem[1].mem = d_mem[0].mem + mem_A;
    d_mem[2].mem = d_mem[1].mem + mem_B;
    d_mem[3].mem = d_mem[2].mem + mem_dst;
    d_mem[4].mem = d_mem[3].mem + mem_A;
    d_mem[5].mem = d_mem[4].mem + mem_B;

    cudaStream_t S[3];
    checkCudaErrors(cudaStreamCreate(&S[0]));
    checkCudaErrors(cudaStreamCreate(&S[1]));
    checkCudaErrors(cudaStreamCreate(&S[2]));

    GEMM3_part_ABC(float, decx::sGEMM_part_ABC)

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(d_mem[0].mem));
    checkCudaErrors(cudaStreamDestroy(S[0]));
    checkCudaErrors(cudaStreamDestroy(S[1]));
    checkCudaErrors(cudaStreamDestroy(S[2]));

    decx::Success(&handle);
    return handle;
}





de::DH de::cuda::GEMM3(MatrixArray<de::Half>& A, MatrixArray<de::Half>& B, MatrixArray<de::Half>& C, MatrixArray<de::Half>& dst)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    _MatrixArray<de::Half>* _A = dynamic_cast<_MatrixArray<de::Half>*>(&A);
    _MatrixArray<de::Half>* _B = dynamic_cast<_MatrixArray<de::Half>*>(&B);
    _MatrixArray<de::Half>* _C = dynamic_cast<_MatrixArray<de::Half>*>(&C);
    _MatrixArray<de::Half>* _dst = dynamic_cast<_MatrixArray<de::Half>*>(&dst);

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A->width % 16) != 0)    ? (_cu_ceil(_A->width, 16) * 16)    : _A->width;
    pitch_B = ((_B->width % 128) != 0)    ? (_cu_ceil(_B->width, 128) * 128)    : _B->width;
    hA        = ((_A->height % 128) != 0) ? (_cu_ceil(_A->height, 128) * 128) : _A->height;

    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    /*
    * [0, 3] -> A;
    * [1, 4] -> B;
    * [2, 5] -> dst;
    */
    decx::alloc::MIF<de::Half> d_mem[6];
    checkCudaErrors(cudaMalloc(&(d_mem[0].mem), (mem_A + mem_B + mem_dst) * 2 * sizeof(de::Half)));
    d_mem[1].mem = d_mem[0].mem + mem_A;
    d_mem[2].mem = d_mem[1].mem + mem_B;
    d_mem[3].mem = d_mem[2].mem + mem_dst;
    d_mem[4].mem = d_mem[3].mem + mem_A;
    d_mem[5].mem = d_mem[4].mem + mem_B;

    cudaStream_t S[3];
    checkCudaErrors(cudaStreamCreate(&S[0]));
    checkCudaErrors(cudaStreamCreate(&S[1]));
    checkCudaErrors(cudaStreamCreate(&S[2]));

    GEMM3_part_ABC(de::Half, decx::hGEMM_part_ABC)

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(d_mem[0].mem));
    checkCudaErrors(cudaStreamDestroy(S[0]));
    checkCudaErrors(cudaStreamDestroy(S[1]));
    checkCudaErrors(cudaStreamDestroy(S[2]));

    decx::Success(&handle);
    return handle;
}