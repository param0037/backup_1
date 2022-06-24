/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once


#include "../../../../core/basic.h"
#include "../../../../classes/core_types.h"
#include "../Conv3_macros.h"


using decx::_Matrix;
using decx::alloc::MIF;
using decx::_MatrixArray;


namespace decx
{
    static void main_loop_hconv2_sk_within8x8_NB(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
            _MatrixArray<de::Half>* src, _Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst,
            MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
            decx::cuda_stream* S[3]);


    static void main_loop_hconv2_sk_exact8x8_NB(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim,
            int2* ker_dim,
            _MatrixArray<de::Half>* src, decx::_Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst,
            MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
            decx::cuda_stream* S[3]);


    static void main_loop_hconv2_sk_within8x16_NB(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        _MatrixArray<de::Half>* src, decx::_Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3]);


    static void main_loop_hconv2_sk_exact8x16_NB(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        _MatrixArray<de::Half>* src, decx::_Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3]);


    static void main_loop_hconv2_sk_within16x8_NB(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        _MatrixArray<de::Half>* src, decx::_Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3]);


    static void main_loop_hconv2_sk_exact16x8_NB(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        _MatrixArray<de::Half>* src, decx::_Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3]);


    static void main_loop_hconv2_sk_within16x16_NB(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        _MatrixArray<de::Half>* src, _Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3]);


    static void main_loop_hconv2_sk_exact16x16_NB(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        _MatrixArray<de::Half>* src, _Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3]);


    // single kernel
    static void _Conv2_NB_R8x8_SK_fp16(_MatrixArray<de::Half>* src, _Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst);


    // single kernel
    static void _Conv2_NB_R8x16_SK_fp16(_MatrixArray<de::Half>* src, _Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst);


    // single kernel
    static void _Conv2_NB_R16x8_SK_fp16(_MatrixArray<de::Half>* src, _Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst);


    // single kernel
    static void _Conv2_NB_R16x16_SK_fp16(_MatrixArray<de::Half>* src, _Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst);


    static void hConv2_border_ignore_sk(_MatrixArray<de::Half>* src, _Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst, de::DH* handle);
}





static void decx::main_loop_hconv2_sk_within8x8_NB(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        _MatrixArray<de::Half>* src, _Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3])
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R8 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R8 - ker_dim->x / 2;

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<de::Half*>(Dmem1->mem) + src_diff.x * Dsrc_alloc_dim->x * 8 + src_diff.y,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(de::Half),
        src->width * sizeof(de::Half),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    // strat the main loop
    main_loop_regulable_R_sk(
        hconv2_kernel_within8x8(Dmem1->mem, Dmem3->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]->get_raw_stream_ptr()),
        hconv2_kernel_within8x8(Dmem2->mem, Dmem4->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]->get_raw_stream_ptr()),
        conv3_main_loop_MemCpyHtoD_within_NB(de::Half, 8),
        de::Half);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




static void decx::main_loop_hconv2_sk_exact8x8_NB(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        _MatrixArray<de::Half>* src, decx::_Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3])
{
    checkCudaErrors(cudaMemcpy2DAsync(Dmem1->mem,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(de::Half),
        src->width * sizeof(de::Half),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    // strat the main loop
    main_loop_regulable_R_sk(
        hconv2_kernel_exact8x8(Dmem1->mem, Dmem3->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]->get_raw_stream_ptr()),
        hconv2_kernel_exact8x8(Dmem2->mem, Dmem4->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]->get_raw_stream_ptr()),
        conv3_main_loop_MemCpyHtoD_exact_NB(de::Half),
        de::Half);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




static void decx::main_loop_hconv2_sk_within8x16_NB(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        _MatrixArray<de::Half>* src, decx::_Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3])
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R8 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R16 - ker_dim->x / 2;

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<de::Half*>(Dmem1->mem) + src_diff.x * Dsrc_alloc_dim->x * 8 + src_diff.y,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(de::Half),
        src->width * sizeof(de::Half),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    // strat the main loop
    main_loop_regulable_R_sk(
        hconv2_kernel_within8x16(Dmem1->mem, Dmem3->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]->get_raw_stream_ptr()),
        hconv2_kernel_within8x16(Dmem2->mem, Dmem4->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]->get_raw_stream_ptr()),
        conv3_main_loop_MemCpyHtoD_within_NB(de::Half, 8),
        de::Half);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}





static void decx::main_loop_hconv2_sk_exact8x16_NB(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        _MatrixArray<de::Half>* src, decx::_Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3])
{
    checkCudaErrors(cudaMemcpy2DAsync(Dmem1->mem,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(de::Half),
        src->width * sizeof(de::Half),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    // strat the main loop
    main_loop_regulable_R_sk(
        hconv2_kernel_exact8x16(Dmem1->mem, Dmem3->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]->get_raw_stream_ptr()),
        hconv2_kernel_exact8x16(Dmem2->mem, Dmem4->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]->get_raw_stream_ptr()),
        conv3_main_loop_MemCpyHtoD_exact_NB(de::Half),
        de::Half);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}





static void decx::main_loop_hconv2_sk_within16x8_NB(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        _MatrixArray<de::Half>* src, decx::_Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3])
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R16 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R8 - ker_dim->x / 2;

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<de::Half*>(Dmem1->mem) + src_diff.x * Dsrc_alloc_dim->x * 8 + src_diff.y,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(de::Half),
        src->width * sizeof(de::Half),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    // strat the main loop
    main_loop_regulable_R_sk(
        hconv2_kernel_within16x8(Dmem1->mem, Dmem3->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]->get_raw_stream_ptr()),
        hconv2_kernel_within16x8(Dmem2->mem, Dmem4->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]->get_raw_stream_ptr()),
        conv3_main_loop_MemCpyHtoD_within_NB(de::Half, 8),
        de::Half);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}





static void decx::main_loop_hconv2_sk_exact16x8_NB(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        _MatrixArray<de::Half>* src, decx::_Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3])
{
    checkCudaErrors(cudaMemcpy2DAsync(Dmem1->mem,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(de::Half),
        src->width * sizeof(de::Half),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    // strat the main loop
    main_loop_regulable_R_sk(
        hconv2_kernel_exact16x8(Dmem1->mem, Dmem3->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]->get_raw_stream_ptr()),
        hconv2_kernel_exact16x8(Dmem2->mem, Dmem4->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]->get_raw_stream_ptr()),
        conv3_main_loop_MemCpyHtoD_exact_NB(de::Half),
        de::Half);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




static void decx::main_loop_hconv2_sk_within16x16_NB(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        _MatrixArray<de::Half>* src, _Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3])
{
    int2 src_diff;
    // copy the first part to device memory
    src_diff.x = bounded_kernel_R16 - ker_dim->y / 2;                src_diff.y = bounded_kernel_R16 - ker_dim->x / 2;

    checkCudaErrors(cudaMemcpy2DAsync(
        reinterpret_cast<de::Half*>(Dmem1->mem) + src_diff.x * Dsrc_alloc_dim->x * 8 + src_diff.y,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(de::Half),
        src->width * sizeof(de::Half),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    // strat the main loop
    main_loop_regulable_R_sk(
        hconv2_kernel_within16x16(Dmem1->mem, Dmem3->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]->get_raw_stream_ptr()),
        hconv2_kernel_within16x16(Dmem2->mem, Dmem4->mem, src_diff, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]->get_raw_stream_ptr()),
        conv3_main_loop_MemCpyHtoD_within_NB(de::Half, 8),
        de::Half)

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




static void decx::main_loop_hconv2_sk_exact16x16_NB(int2* Dsrc_alloc_dim, int2* Ddst_alloc_dim, int2* ker_dim,
        _MatrixArray<de::Half>* src, _Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst,
        MIF<float4>* Dmem1, MIF<float4>* Dmem2, MIF<float4>* Dmem3, MIF<float4>* Dmem4,
        decx::cuda_stream* S[3])
{
    checkCudaErrors(cudaMemcpy2DAsync(Dmem1->mem,
        Dsrc_alloc_dim->x * sizeof(float4),
        src->MatptrArr.ptr[0],
        src->pitch * sizeof(de::Half),
        src->width * sizeof(de::Half),
        src->height,
        cudaMemcpyHostToDevice,
        S[0]->get_raw_stream_ref()));                            // copy the datas of src from host to device
    Dmem1->leading = true;
    Dmem1->_using = false;

    checkCudaErrors(cudaDeviceSynchronize());

    // strat the main loop
    main_loop_regulable_R_sk(
        hconv2_kernel_exact16x16(Dmem1->mem, Dmem3->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]->get_raw_stream_ptr()),
        hconv2_kernel_exact16x16(Dmem2->mem, Dmem4->mem, *Dsrc_alloc_dim, *Ddst_alloc_dim, *ker_dim, S[0]->get_raw_stream_ptr()),
        conv3_main_loop_MemCpyHtoD_exact_NB(de::Half),
        de::Half);

    if (Dmem3->leading) {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem3->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem3->_using = true;
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(dst->MatptrArr.ptr[dst->ArrayNumber - 1],
            dst->pitch * sizeof(de::Half), Dmem4->mem, Ddst_alloc_dim->x * sizeof(float4), dst->width * sizeof(de::Half),
            dst->height, cudaMemcpyDeviceToHost, S[2]->get_raw_stream_ref()));
        Dmem4->_using = false;
    }

    checkCudaErrors(cudaDeviceSynchronize());
}




// **************************************************************************************************************************


// single kernel
static void decx::_Conv2_NB_R8x8_SK_fp16(_MatrixArray<de::Half>* src, _Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst)
{
    MIF<float4> Dmem1, Dmem2,    // for src
        Dmem3, Dmem4;            // for dst

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 128) * bounded_kernel_R8 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R8 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R8 / 4;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R8 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    checkCudaErrors(cudaMalloc(&Dmem1.mem, 2 * (dev_src_size + dev_dst_size) * sizeof(float4)));

    Dmem2.mem = Dmem1.mem + dev_src_size;
    Dmem3.mem = Dmem2.mem + dev_src_size;
    Dmem4.mem = Dmem3.mem + dev_dst_size;

    decx::cuda_stream* S[3];
    for (int i = 0; i < 3; ++i) {
        S[i] = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    }

    uint offset_lin = 0, offset_ker = 0;

    for (int k = 0; k < kernel->height; ++k) {
        cudaMemcpyToSymbolAsync(Const_Mem, kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(de::Half), offset_lin * sizeof(de::Half), cudaMemcpyHostToDevice, S[0]->get_raw_stream_ref());
        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }
    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R8 * 2 + 1) && ker_dim.y == (bounded_kernel_R8 * 2 + 1)) {
        main_loop_hconv2_sk_exact8x8_NB(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S);
    }
    else {
        main_loop_hconv2_sk_within8x8_NB(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S);
    }

    checkCudaErrors(cudaFree(Dmem1.mem));
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i]->detach();
    }
}




// single kernel
static void decx::_Conv2_NB_R8x16_SK_fp16(_MatrixArray<de::Half>* src, _Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst)
{
    MIF<float4> Dmem1, Dmem2,    // for src
        Dmem3, Dmem4;            // for dst

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 128) * bounded_kernel_R8 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R8 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R16 / 4;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R8 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    checkCudaErrors(cudaMalloc(&Dmem1.mem, 2 * (dev_src_size + dev_dst_size) * sizeof(float4)));

    Dmem2.mem = Dmem1.mem + dev_src_size;
    Dmem3.mem = Dmem2.mem + dev_src_size;
    Dmem4.mem = Dmem3.mem + dev_dst_size;

    decx::cuda_stream* S[3];
    for (int i = 0; i < 3; ++i) {
        S[i] = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    }

    uint offset_lin = 0, offset_ker = 0;

    for (int k = 0; k < kernel->height; ++k) {
        cudaMemcpyToSymbolAsync(Const_Mem, kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(de::Half), offset_lin * sizeof(de::Half), cudaMemcpyHostToDevice, S[0]->get_raw_stream_ref());
        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }
    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R16 * 2 + 1) && ker_dim.y == (bounded_kernel_R8 * 2 + 1)) {
        main_loop_hconv2_sk_exact8x16_NB(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S);
    }
    else {
        main_loop_hconv2_sk_within8x16_NB(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S);
    }

    checkCudaErrors(cudaFree(Dmem1.mem));
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i]->detach();
    }
}




// single kernel
static void decx::_Conv2_NB_R16x8_SK_fp16(_MatrixArray<de::Half>* src, _Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst)
{
    MIF<float4> Dmem1, Dmem2,    // for src
        Dmem3, Dmem4;            // for dst

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 128) * bounded_kernel_R8 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R8 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R8 / 4;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R16 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    checkCudaErrors(cudaMalloc(&Dmem1.mem, 2 * (dev_src_size + dev_dst_size) * sizeof(float4)));

    Dmem2.mem = Dmem1.mem + dev_src_size;
    Dmem3.mem = Dmem2.mem + dev_src_size;
    Dmem4.mem = Dmem3.mem + dev_dst_size;

    decx::cuda_stream* S[3];
    for (int i = 0; i < 3; ++i) {
        S[i] = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    }

    uint offset_lin = 0, offset_ker = 0;

    for (int k = 0; k < kernel->height; ++k) {
        cudaMemcpyToSymbolAsync(Const_Mem, kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(de::Half), offset_lin * sizeof(de::Half), cudaMemcpyHostToDevice, S[0]->get_raw_stream_ref());
        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }
    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R8 * 2 + 1) && ker_dim.y == (bounded_kernel_R16 * 2 + 1)) {
        main_loop_hconv2_sk_exact16x8_NB(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S);
    }
    else {
        main_loop_hconv2_sk_within16x8_NB(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S);
    }

    checkCudaErrors(cudaFree(Dmem1.mem));
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i]->detach();
    }
}




// single kernel
static void decx::_Conv2_NB_R16x16_SK_fp16(_MatrixArray<de::Half>* src, _Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst)
{
    MIF<float4> Dmem1, Dmem2,    // for src
        Dmem3, Dmem4;            // for dst

    int2 Dsrc_alloc_dim,
        Ddst_alloc_dim,
        ker_dim;

    ker_dim.x = kernel->width;            ker_dim.y = kernel->height;
    // first, allocate the memory according to R8 alignments
    Ddst_alloc_dim.x = decx::utils::ceil<int>(dst->width, 128) * bounded_kernel_R8 * 2;
    Ddst_alloc_dim.y = decx::utils::ceil<int>(dst->height, 16) * bounded_kernel_R8 * 2;

    Dsrc_alloc_dim.x = Ddst_alloc_dim.x + bounded_kernel_R16 / 4;        // bounded_kernel_R8 * 2 / 4
    Dsrc_alloc_dim.y = Ddst_alloc_dim.y + bounded_kernel_R16 * 2;

    size_t dev_src_size = static_cast<size_t>(Dsrc_alloc_dim.x) * static_cast<size_t>(Dsrc_alloc_dim.y);
    size_t dev_dst_size = static_cast<size_t>(Ddst_alloc_dim.x) * static_cast<size_t>(Ddst_alloc_dim.y);

    checkCudaErrors(cudaMalloc(&Dmem1.mem, 2 * (dev_src_size + dev_dst_size) * sizeof(float4)));

    Dmem2.mem = Dmem1.mem + dev_src_size;
    Dmem3.mem = Dmem2.mem + dev_src_size;
    Dmem4.mem = Dmem3.mem + dev_dst_size;

    decx::cuda_stream* S[3];
    for (int i = 0; i < 3; ++i) {
        S[i] = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    }

    uint offset_lin = 0, offset_ker = 0;

    for (int k = 0; k < kernel->height; ++k) {
        cudaMemcpyToSymbolAsync(Const_Mem, kernel->Mat.ptr + offset_ker,
            kernel->width * sizeof(de::Half), offset_lin * sizeof(de::Half), cudaMemcpyHostToDevice, S[0]->get_raw_stream_ref());
        offset_lin += kernel->width;
        offset_ker += kernel->pitch;
    }
    // strat the main loop
    if (ker_dim.x == (bounded_kernel_R16 * 2 + 1) && ker_dim.y == (bounded_kernel_R16 * 2 + 1)) {
        main_loop_hconv2_sk_exact16x16_NB(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S);
    }
    else {
        main_loop_hconv2_sk_within16x16_NB(
            &Dsrc_alloc_dim, &Ddst_alloc_dim, &ker_dim, src, kernel, dst, &Dmem1, &Dmem2, &Dmem3, &Dmem4, S);
    }

    checkCudaErrors(cudaFree(Dmem1.mem));
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        S[i]->detach();
    }
}





// ******************************************************************************************************************************

static void decx::hConv2_border_ignore_sk(_MatrixArray<de::Half>* src, _Matrix<de::Half>* kernel, _MatrixArray<de::Half>* dst, de::DH* handle)
{
    int2 half_ker_dim;
    half_ker_dim.x = kernel->width / 2;                half_ker_dim.y = kernel->height / 2;

    const uint4 dst_dim = make_uint4(src->width - (half_ker_dim.x * 2),
                                     src->height - (half_ker_dim.y * 2),
                                     src->ArrayNumber,
                                     decx::DATA_STORE_TYPE::Page_Locked);

    decx::conv2_mc_dst_rearrangement(dst, dst_dim);

    if (half_ker_dim.x < bounded_kernel_R8 + 1) {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            _Conv2_NB_R8x8_SK_fp16(src, kernel, dst);
        }
        else {
            _Conv2_NB_R16x8_SK_fp16(src, kernel, dst);
        }
    }
    else {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            _Conv2_NB_R8x16_SK_fp16(src, kernel, dst);
        }
        else {
            _Conv2_NB_R16x16_SK_fp16(src, kernel, dst);
        }
    }
    decx::Success(handle);
}