/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#ifndef _SGEMM_CALLERS_H
#define _SGEMM_CALLERS_H

#include "gemm_utils.h"
#include "sgemm_calc_kernel.h"


namespace decx
{
    /*
    * dims_info.x -> global_wA (linear)
    * dims_info.y -> global_wB (8x satisfied) = wC
    */
    static void _THREAD_FUNCTION_ _ST_sgemm_Dblock_FH_FL_W16(
        float* A, float* B, float* C, const int __linear, const int global_wB, const int loc_hA, const int loc_wB);

    /*
    * dims_info.x -> global_wA (linear)
    * dims_info.y -> global_wB (8x satisfied) = wC
    */
    static void _THREAD_FUNCTION_ _ST_sgemm_Dblock_LH_FL_W16(
        float* A, float* B, float* C, const int __linear, const int global_wB, const int loc_hA, const int loc_wB);


    /*
    * dims_info.x -> global_wA (linear)
    * dims_info.y -> global_wB (8x satisfied) = wC
    */
    static void _THREAD_FUNCTION_ _ST_sgemm_Dblock_FH_LL_W16(
        float* A, float* B, float* C, const int __linear, const int global_wB, const int loc_hA, const int loc_wB);


    /*
    * dims_info.x -> global_wA (linear)
    * dims_info.y -> global_wB (8x satisfied) = wC
    */
    static void _THREAD_FUNCTION_ _ST_sgemm_Dblock_LH_LL_W16(
        float* A, float* B, float* C, const int __linear, const int global_wB, const int loc_hA, const int loc_wB);

    /*
    * dims_info.x -> global_wA (linear)
    * dims_info.y -> global_wB (8x satisfied) = wC
    */
    static void _THREAD_FUNCTION_ _ST_sgemm_Dblock_FH_FL_W8(
        float* A, float* B, float* C, const int __linear, const int global_wB, const int loc_hA, const int loc_wB);

    /*
    * dims_info.x -> global_wA (linear)
    * dims_info.y -> global_wB (8x satisfied) = wC
    */
    static void _THREAD_FUNCTION_ _ST_sgemm_Dblock_FH_LL_W8(
        float* A, float* B, float* C, const int __linear, const int global_wB, const int loc_hA, const int loc_wB);

    /*
    * dims_info.x -> global_wA (linear)
    * dims_info.y -> global_wB (8x satisfied) = wC
    */
    static void _THREAD_FUNCTION_ _ST_sgemm_Dblock_LH_LL_W8(
        float* A, float* B, float* C, const int __linear, const int global_wB, const int loc_hA, const int loc_wB);

    /*
    * dims_info.x -> global_wA (linear)
    * dims_info.y -> global_wB (8x satisfied) = wC
    */
    static void _THREAD_FUNCTION_ _ST_sgemm_Dblock_LH_FL_W8(
        float* A, float* B, float* C, const int __linear, const int global_wB, const int loc_hA, const int loc_wB);


    /**
    * @param : glo_dims.x -> __linear (wA) (hB) .y -> global_hA .z -> global_wB (16x) .w -> 缺省
    * @param : proc_dim.x -> local_hA .y -> local_wB
    */
    template <int threadDim_x, int threadDim_y>
    static void sgemm_Dblock_FH_FL_FW(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim, const size_t& fake_wB);


    /**
    * @param : glo_dims.x -> __linear (wA) (hB) .y -> global_hA .z -> global_wB (16x) .w -> 缺省
    * @param : proc_dim.x -> local_hA .y -> local_wB
    */
    template <int threadDim_x, int threadDim_y>
    static void sgemm_Dblock_FH_LL_FW(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim, const size_t& fake_wB);



    /**
    * @param : glo_dims.x -> __linear (wA) (hB) .y -> global_hA .z -> global_wB (16x) .w -> 缺省
    * @param : proc_dim.x -> local_hA .y -> local_wB
    */
    template <int threadDim_x, int threadDim_y>
    static void sgemm_Dblock_FH_FL_LW16(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
        const int wB_left, const size_t& fake_wB);


    /**
    * @param : glo_dims.x -> __linear (wA) (hB) .y -> global_hA .z -> global_wB (16x) .w -> 缺省
    * @param : proc_dim.x -> local_hA .y -> local_wB
    */
    template <int threadDim_x, int threadDim_y>
    static void sgemm_Dblock_FH_LL_LW16(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
        const int wB_left, const size_t& fake_wB);

    /**
    * @param : glo_dims.x -> __linear (wA) (hB) .y -> global_hA .z -> global_wB (16x) .w -> 缺省
    * @param : proc_dim.x -> local_hA .y -> local_wB
    */
    template <int threadDim_x, int threadDim_y>
    static void sgemm_Dblock_FH_FL_LW8(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
        const int wB_left, const size_t& fake_wB);

    /**
    * @param : glo_dims.x -> __linear (wA) (hB) .y -> global_hA .z -> global_wB (16x) .w -> 缺省
    * @param : proc_dim.x -> local_hA .y -> local_wB
    */
    template <int threadDim_x, int threadDim_y>
    static void sgemm_Dblock_FH_LL_LW8(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
        const int wB_left, const size_t& fake_wB);

    /**
    * @param : glo_dims.x -> __linear (wA) (hB) .y -> global_hA .z -> global_wB (16x) .w -> 缺省
    * @param : proc_dim.x -> local_hA .y -> local_wB
    */
    template <int threadDim_x, int threadDim_y>
    static void sgemm_Dblock_LH_FL_FW(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
        const int hA_left, const size_t& fake_wB);

    /**
    * @param : glo_dims.x -> __linear (wA) (hB) .y -> global_hA .z -> global_wB (16x) .w -> 缺省
    * @param : proc_dim.x -> local_hA .y -> local_wB
    */
    template <int threadDim_x, int threadDim_y>
    static void sgemm_Dblock_LH_FL_LW16(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
        const int wB_left, const int hA_left, const size_t& fake_wB);


    template <int threadDim_x, int threadDim_y>
    static void sgemm_Dblock_LH_LL_FW(float* A, float* B, float* C, const int4* glo_dim,
        const int2* proc_dim, const int hA_left, const size_t& fake_wB);

    template <int threadDim_x, int threadDim_y>
    static void sgemm_Dblock_LH_LL_LW16(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
        const int wB_left, const int hA_left, const size_t& fake_wB);

    template <int threadDim_x, int threadDim_y>
    static void sgemm_Dblock_LH_FL_LW8(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
        const int wB_left, const int hA_left, const size_t& fake_wB);

    template <int threadDim_x, int threadDim_y>
    static void sgemm_Dblock_LH_LL_LW8(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
        const int wB_left, const int hA_left, const size_t& fake_wB);


    /**
    * @param dims_pkg : (in __m128).x -> __linear (wA) (hB) .y -> hA .z -> wB (16x) .w -> 缺省
    */
    template <int threadDim_x, int threadDim_y>
    static void sgemm_caller_FH_W16(float* A, float* B, float* C, const int4* glo_dim);


    /**
    * @param dims_pkg : (in __m128).x -> __linear (wA) (hB) .y -> hA .z -> wB (16x) .w -> 缺省
    */
    template <int threadDim_x, int threadDim_y>
    static void sgemm_caller_LH_W16(float* A, float* B, float* C, const int4* glo_dim);


    /**
    * @param dims_pkg : (in __m128).x -> __linear (wA) (hB) .y -> hA .z -> wB (16x) .w -> 缺省
    */
    template <int threadDim_x, int threadDim_y>
    static void sgemm_caller_FH_W8(float* A, float* B, float* C, const int4* glo_dim);


    /**
    * @param dims_pkg : (in __m128).x -> __linear (wA) (hB) .y -> hA .z -> wB (16x) .w -> 缺省
    */
    template <int threadDim_x, int threadDim_y>
    static void sgemm_caller_LH_W8(float* A, float* B, float* C, const int4* glo_dim);


    /**
    * @param A : pointer of matrix A
    * @param B : pointer of matrix B
    * @param tmp_B : pointer of matrix tmp_B
    * @param dst : pointer of matrix dst
    * @param dims_pkg : (in __m128).x -> __linear (wA) (hB) .y -> hA .z -> wB (16x) .w -> 缺省
    */
    template <int threadDim_x, int threadDim_y>
    static void sgemm_caller(float* A, float* B, float* tmp_B, float* dst, const int4* glo_dim);
}



#define AVX2
#define M_size 1024

// __linear 区域需要是4的倍数，hA无要求，wB需要是16的倍数
// 每个thread有8192个float的Lcache空间
// fragA(64, 32), fragB(32, 64), fragC(32, 32) -> 2048 + 2048 + 1024 = 5120 < 8192

typedef unsigned char uchar;

#define sgemm_BL_Linear 64      // 8次迭代
#define sgemm_BL_hA 32          // 32次迭代
#define sgemm_BL_wB 16          // 2次迭代
#define sgemm_BL_linear_iter 8  // = sgemm_BL_Linear / 4(length of __m128)



static void _THREAD_FUNCTION_ decx::_ST_sgemm_Dblock_FH_FL_W16(
    float* A, float* B, float* C, const int __linear, const int global_wB, const int loc_hA, const int loc_wB)
{
    size_t A_dex = 0, B_dex = 0, C_dex = 0,
        tmp_A;
    constexpr size_t B_step = 16 * sgemm_BL_Linear;

    for (int i = 0; i < loc_hA; i += sgemm_BL_hA)
    {
        B_dex = 0;
        for (int j = 0; j < loc_wB; j += sgemm_BL_wB) {
            tmp_A = A_dex;
            decx::_avx256_sgemm_block_first(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
            B_dex += B_step;
            tmp_A += sgemm_BL_Linear;

            for (int k = sgemm_BL_Linear; k < __linear; k += sgemm_BL_Linear) {
                decx::_avx256_sgemm_block(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
                B_dex += B_step;
                tmp_A += sgemm_BL_Linear;
            }
            C_dex += sgemm_BL_wB;
        }
        C_dex += sgemm_BL_hA * (size_t)global_wB - (size_t)loc_wB;
        A_dex += sgemm_BL_hA * (size_t)__linear;
    }
}



static void _THREAD_FUNCTION_ decx::_ST_sgemm_Dblock_LH_FL_W16(
    float* A, float* B, float* C, const int __linear, const int global_wB, const int loc_hA, const int loc_wB)
{
    size_t A_dex = 0, B_dex = 0, C_dex = 0,
        tmp_A;
    constexpr size_t B_step = 16 * sgemm_BL_Linear;

    for (int i = 0; i < loc_hA / sgemm_BL_hA; ++i)
    {
        B_dex = 0;
        for (int j = 0; j < loc_wB; j += sgemm_BL_wB) {
            tmp_A = A_dex;
            decx::_avx256_sgemm_block_first(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
            B_dex += B_step;
            tmp_A += sgemm_BL_Linear;

            for (int k = sgemm_BL_Linear; k < __linear; k += sgemm_BL_Linear) {
                decx::_avx256_sgemm_block(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
                B_dex += B_step;
                tmp_A += sgemm_BL_Linear;
            }
            C_dex += sgemm_BL_wB;
        }
        C_dex += sgemm_BL_hA * (size_t)global_wB - (size_t)loc_wB;
        A_dex += sgemm_BL_hA * (size_t)__linear;
    }
    // left height
    B_dex = 0;
    for (int j = 0; j < loc_wB; j += sgemm_BL_wB) {
        tmp_A = A_dex;
        decx::_avx256_sgemm_block_flexibleH_first(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, loc_hA % sgemm_BL_hA);
        B_dex += B_step;
        tmp_A += sgemm_BL_Linear;

        for (int k = sgemm_BL_Linear; k < __linear; k += sgemm_BL_Linear) {
            decx::_avx256_sgemm_block_flexibleH(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, loc_hA % sgemm_BL_hA);
            B_dex += B_step;
            tmp_A += sgemm_BL_Linear;
        }
        C_dex += sgemm_BL_wB;
    }
}



static void _THREAD_FUNCTION_ decx::_ST_sgemm_Dblock_FH_LL_W16(
    float* A, float* B, float* C, const int __linear, const int global_wB, const int loc_hA, const int loc_wB)
{
    size_t A_dex = 0, B_dex = 0, C_dex = 0, tmp_A;
    int _Lleft = __linear % sgemm_BL_Linear;
    constexpr size_t B_step = 16 * sgemm_BL_Linear;
    size_t B_gap = 16 * (size_t)_Lleft;

    for (int i = 0; i < loc_hA; i += sgemm_BL_hA)
    {
        B_dex = 0;
        for (int j = 0; j < loc_wB; j += sgemm_BL_wB) {
            tmp_A = A_dex;
            decx::_avx256_sgemm_block_first(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
            B_dex += B_step;
            tmp_A += sgemm_BL_Linear;

            for (int k = 1; k < __linear / sgemm_BL_Linear; ++k) {
                decx::_avx256_sgemm_block(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
                B_dex += B_step;
                tmp_A += sgemm_BL_Linear;
            }
            decx::_avx256_sgemm_block_Lleft(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, _Lleft);
            tmp_A += sgemm_BL_Linear;

            C_dex += sgemm_BL_wB;
            B_dex += B_gap;
        }
        C_dex += sgemm_BL_hA * (size_t)global_wB - (size_t)loc_wB;
        A_dex += sgemm_BL_hA * (size_t)__linear;
    }
}



static void _THREAD_FUNCTION_ decx::_ST_sgemm_Dblock_LH_LL_W16(
    float* A, float* B, float* C, const int __linear, const int global_wB, const int loc_hA, const int loc_wB)
{
    size_t A_dex = 0, B_dex = 0, C_dex = 0,
        tmp_A;
    constexpr size_t B_step = 16 * sgemm_BL_Linear;
    int _Lleft = __linear % sgemm_BL_Linear;
    size_t B_gap = 16 * (size_t)_Lleft;

    for (int i = 0; i < loc_hA / sgemm_BL_hA; ++i)
    {
        B_dex = 0;
        for (int j = 0; j < loc_wB; j += sgemm_BL_wB) {
            tmp_A = A_dex;
            decx::_avx256_sgemm_block_first(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
            B_dex += B_step;
            tmp_A += sgemm_BL_Linear;

            for (int k = 1; k < __linear / sgemm_BL_Linear; ++k) {
                decx::_avx256_sgemm_block(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
                B_dex += B_step;
                tmp_A += sgemm_BL_Linear;
            }
            decx::_avx256_sgemm_block_Lleft(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, _Lleft);
            tmp_A += sgemm_BL_Linear;

            C_dex += sgemm_BL_wB;
            B_dex += B_gap;
        }
        C_dex += sgemm_BL_hA * (size_t)global_wB - (size_t)loc_wB;
        A_dex += sgemm_BL_hA * (size_t)__linear;
    }
    B_dex = 0;
    for (int j = 0; j < loc_wB; j += sgemm_BL_wB) {
        tmp_A = A_dex;
        decx::_avx256_sgemm_block_flexibleH_first(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, loc_hA % sgemm_BL_hA);
        B_dex += B_step;
        tmp_A += sgemm_BL_Linear;

        for (int k = 1; k < __linear / sgemm_BL_Linear; ++k) {
            decx::_avx256_sgemm_block_flexibleH(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, loc_hA % sgemm_BL_hA);
            B_dex += B_step;
            tmp_A += sgemm_BL_Linear;
        }
        decx::_avx256_sgemm_block_Lleft_flexibleH(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, _Lleft, loc_hA % sgemm_BL_hA);
        tmp_A += sgemm_BL_Linear;

        C_dex += sgemm_BL_wB;
        B_dex += B_gap;
    }
}



static void _THREAD_FUNCTION_ decx::_ST_sgemm_Dblock_FH_FL_W8(
    float* A, float* B, float* C, const int __linear, const int global_wB, const int loc_hA, const int loc_wB)
{
    size_t A_dex = 0, B_dex = 0, C_dex = 0, tmp_A;
    constexpr size_t B_step = 16 * sgemm_BL_Linear;
    constexpr size_t B_step_8 = 8 * sgemm_BL_Linear;

    for (int i = 0; i < loc_hA; i += sgemm_BL_hA)
    {
        B_dex = 0;
        for (int j = 0; j < loc_wB - 8; j += sgemm_BL_wB) {
            tmp_A = A_dex;
            decx::_avx256_sgemm_block_first(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
            B_dex += B_step;
            tmp_A += sgemm_BL_Linear;

            for (int k = sgemm_BL_Linear; k < __linear; k += sgemm_BL_Linear) {
                decx::_avx256_sgemm_block(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
                B_dex += B_step;
                tmp_A += sgemm_BL_Linear;
            }
            C_dex += sgemm_BL_wB;
        }
        tmp_A = A_dex;
        decx::_avx256_sgemm_block_first_w8(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
        B_dex += B_step_8;
        tmp_A += sgemm_BL_Linear;

        for (int k = sgemm_BL_Linear; k < __linear; k += sgemm_BL_Linear) {
            decx::_avx256_sgemm_block_w8(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
            B_dex += B_step_8;
            tmp_A += sgemm_BL_Linear;
        }

        C_dex += ((size_t)global_wB << 5) - (size_t)loc_wB + 8;
        A_dex += ((size_t)__linear << 5);
    }
}



static void _THREAD_FUNCTION_ decx::_ST_sgemm_Dblock_FH_LL_W8(
    float* A, float* B, float* C, const int __linear, const int global_wB, const int loc_hA, const int loc_wB)
{
    size_t A_dex = 0, B_dex = 0, C_dex = 0,
        tmp_A;

    int _Lleft = __linear % sgemm_BL_Linear;

    size_t B_gap = 16 * (size_t)_Lleft;
    for (int i = 0; i < loc_hA; i += sgemm_BL_hA)
    {
        B_dex = 0;
        for (int j = 0; j < loc_wB - 8; j += sgemm_BL_wB) {
            tmp_A = A_dex;
            decx::_avx256_sgemm_block_first(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
            B_dex += 16 * sgemm_BL_Linear;
            tmp_A += sgemm_BL_Linear;

            for (int k = 1; k < __linear / sgemm_BL_Linear; ++k) {
                decx::_avx256_sgemm_block(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
                B_dex += 16 * sgemm_BL_Linear;
                tmp_A += sgemm_BL_Linear;
            }
            decx::_avx256_sgemm_block_Lleft(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, _Lleft);
            tmp_A += sgemm_BL_Linear;

            C_dex += sgemm_BL_wB;
            B_dex += B_gap;
        }

        tmp_A = A_dex;
        decx::_avx256_sgemm_block_first_w8(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
        B_dex += 8 * sgemm_BL_Linear;
        tmp_A += sgemm_BL_Linear;

        for (int k = 1; k < __linear / sgemm_BL_Linear; ++k) {
            decx::_avx256_sgemm_block_w8(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
            B_dex += 8 * sgemm_BL_Linear;
            tmp_A += sgemm_BL_Linear;
        }
        decx::_avx256_sgemm_block_Lleft_w8(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, _Lleft);
        tmp_A += sgemm_BL_Linear;

        C_dex += ((size_t)global_wB << 5) - (size_t)loc_wB + 8;
        A_dex += ((size_t)__linear << 5);
    }
}



static void _THREAD_FUNCTION_ decx::_ST_sgemm_Dblock_LH_LL_W8(
    float* A, float* B, float* C, const int __linear, const int global_wB, const int loc_hA, const int loc_wB)
{
    size_t A_dex = 0, B_dex = 0, C_dex = 0,
        tmp_A;

    int _Lleft = __linear % sgemm_BL_Linear;

    size_t B_gap = 16 * (size_t)_Lleft;
    for (int i = 0; i < loc_hA / sgemm_BL_hA; ++i)
    {
        B_dex = 0;
        for (int j = 0; j < loc_wB - 8; j += sgemm_BL_wB) {
            tmp_A = A_dex;
            decx::_avx256_sgemm_block_first(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
            B_dex += 16 * sgemm_BL_Linear;
            tmp_A += sgemm_BL_Linear;

            for (int k = 1; k < __linear / sgemm_BL_Linear; ++k) {
                decx::_avx256_sgemm_block(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
                B_dex += 16 * sgemm_BL_Linear;
                tmp_A += sgemm_BL_Linear;
            }
            decx::_avx256_sgemm_block_Lleft(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, _Lleft);
            tmp_A += sgemm_BL_Linear;

            C_dex += sgemm_BL_wB;
            B_dex += B_gap;
        }

        tmp_A = A_dex;
        decx::_avx256_sgemm_block_first_w8(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
        B_dex += 8 * sgemm_BL_Linear;
        tmp_A += sgemm_BL_Linear;

        for (int k = 1; k < __linear / sgemm_BL_Linear; ++k) {
            decx::_avx256_sgemm_block_w8(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
            B_dex += 8 * sgemm_BL_Linear;
            tmp_A += sgemm_BL_Linear;
        }
        decx::_avx256_sgemm_block_Lleft_w8(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, _Lleft);
        tmp_A += sgemm_BL_Linear;

        C_dex += 8;

        C_dex += sgemm_BL_hA * (size_t)global_wB - (size_t)loc_wB;
        A_dex += sgemm_BL_hA * (size_t)__linear;
    }
    // left height
    B_dex = 0;
    for (int j = 0; j < loc_wB - 8; j += sgemm_BL_wB) {
        tmp_A = A_dex;
        decx::_avx256_sgemm_block_flexibleH_first(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, loc_hA % sgemm_BL_hA);
        B_dex += 16 * sgemm_BL_Linear;
        tmp_A += sgemm_BL_Linear;

        for (int k = 1; k < __linear / sgemm_BL_Linear; ++k) {
            decx::_avx256_sgemm_block_flexibleH(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, loc_hA % sgemm_BL_hA);
            B_dex += 16 * sgemm_BL_Linear;
            tmp_A += sgemm_BL_Linear;
        }
        decx::_avx256_sgemm_block_Lleft_flexibleH(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, _Lleft, loc_hA % sgemm_BL_hA);
        tmp_A += sgemm_BL_Linear;

        C_dex += sgemm_BL_wB;
        B_dex += B_gap;
    }

    tmp_A = A_dex;
    decx::_avx256_sgemm_block_first_flexibleH_w8(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, loc_hA % sgemm_BL_hA);
    B_dex += 8 * sgemm_BL_Linear;
    tmp_A += sgemm_BL_Linear;

    for (int k = 1; k < __linear / sgemm_BL_Linear; ++k) {
        decx::_avx256_sgemm_block_flexibleH_w8(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, loc_hA % sgemm_BL_hA);
        B_dex += 8 * sgemm_BL_Linear;
        tmp_A += sgemm_BL_Linear;
    }
    decx::_avx256_sgemm_block_Lleft_flexibleH_w8(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, _Lleft, loc_hA % sgemm_BL_hA);
    tmp_A += sgemm_BL_Linear;

    C_dex += 8;

    C_dex += sgemm_BL_hA * (size_t)global_wB - (size_t)loc_wB;
    A_dex += sgemm_BL_hA * (size_t)__linear;
}



static void _THREAD_FUNCTION_ decx::_ST_sgemm_Dblock_LH_FL_W8(
    float* A, float* B, float* C, const int __linear, const int global_wB, const int loc_hA, const int loc_wB)
{
    size_t A_dex = 0, B_dex = 0, C_dex = 0,
        tmp_A;

    int _Lleft = __linear % sgemm_BL_Linear;

    size_t B_gap = 16 * (size_t)_Lleft;
    for (int i = 0; i < loc_hA / sgemm_BL_hA; ++i)
    {
        B_dex = 0;
        for (int j = 0; j < loc_wB - 8; j += sgemm_BL_wB) {
            tmp_A = A_dex;
            decx::_avx256_sgemm_block_first(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
            B_dex += 16 * sgemm_BL_Linear;
            tmp_A += sgemm_BL_Linear;

            for (int k = 1; k < __linear / sgemm_BL_Linear; ++k) {
                decx::_avx256_sgemm_block(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
                B_dex += 16 * sgemm_BL_Linear;
                tmp_A += sgemm_BL_Linear;
            }
            
            C_dex += sgemm_BL_wB;
            B_dex += B_gap;
        }

        tmp_A = A_dex;
        decx::_avx256_sgemm_block_first_w8(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
        B_dex += 8 * sgemm_BL_Linear;
        tmp_A += sgemm_BL_Linear;

        for (int k = 1; k < __linear / sgemm_BL_Linear; ++k) {
            decx::_avx256_sgemm_block_w8(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB);
            B_dex += 8 * sgemm_BL_Linear;
            tmp_A += sgemm_BL_Linear;
        }
        
        C_dex += sgemm_BL_hA * (size_t)global_wB - (size_t)loc_wB + 8;
        A_dex += sgemm_BL_hA * (size_t)__linear;
    }
    // left height
    B_dex = 0;
    for (int j = 0; j < loc_wB - 8; j += sgemm_BL_wB) {
        tmp_A = A_dex;
        decx::_avx256_sgemm_block_flexibleH_first(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, loc_hA % sgemm_BL_hA);
        B_dex += 16 * sgemm_BL_Linear;
        tmp_A += sgemm_BL_Linear;

        for (int k = 1; k < __linear / sgemm_BL_Linear; ++k) {
            decx::_avx256_sgemm_block_flexibleH(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, loc_hA % sgemm_BL_hA);
            B_dex += 16 * sgemm_BL_Linear;
            tmp_A += sgemm_BL_Linear;
        }
        
        C_dex += sgemm_BL_wB;
        B_dex += B_gap;
    }

    tmp_A = A_dex;
    decx::_avx256_sgemm_block_first_flexibleH_w8(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, loc_hA % sgemm_BL_hA);
    B_dex += 8 * sgemm_BL_Linear;
    tmp_A += sgemm_BL_Linear;

    for (int k = 1; k < __linear / sgemm_BL_Linear; ++k) {
        decx::_avx256_sgemm_block_flexibleH_w8(A + tmp_A, B + B_dex, C + C_dex, __linear, global_wB, loc_hA % sgemm_BL_hA);
        B_dex += 8 * sgemm_BL_Linear;
        tmp_A += sgemm_BL_Linear;
    }
    
    C_dex += sgemm_BL_hA * (size_t)global_wB - (size_t)loc_wB + 8;
    A_dex += sgemm_BL_hA * (size_t)__linear;
}



// --------------------------------------------------------------------------------------------------------



template <int threadDim_x, int threadDim_y>
void decx::sgemm_Dblock_FH_FL_FW(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim, const size_t& fake_wB)
{
    std::future<void> __async_stream[threadDim_x * threadDim_y];

    int sum = 0;
    for (int i = 0; i < threadDim_x; ++i) {
        for (int j = 0; j < threadDim_y; ++j) {
            __async_stream[sum] = decx::thread_pool.register_task(
                decx::_ST_sgemm_Dblock_FH_FL_W16,
                A + i * (size_t)proc_dim->x * (size_t)glo_dim->x,
                B + j * (size_t)fake_wB,
                C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
                glo_dim->x, glo_dim->z, proc_dim->x, proc_dim->y);
            ++sum;
        }
    }

    for (int i = 0; i < threadDim_x * threadDim_y; ++i) {
        __async_stream[i].get();
    }
}



template <int threadDim_x, int threadDim_y>
void decx::sgemm_Dblock_FH_LL_FW(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim, const size_t& fake_wB)
{
    std::future<void> __async_stream[threadDim_x * threadDim_y];

    int sum = 0;
    for (int i = 0; i < threadDim_x; ++i) {
        for (int j = 0; j < threadDim_y; ++j) {
            __async_stream[sum] = decx::thread_pool.register_task(
                decx::_ST_sgemm_Dblock_FH_LL_W16,
                A + i * (size_t)proc_dim->x * (size_t)glo_dim->x,
                B + j * (size_t)fake_wB,
                C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
                glo_dim->x, glo_dim->z, proc_dim->x, proc_dim->y);
            ++sum;
        }
    }

    for (int i = 0; i < threadDim_x * threadDim_y; ++i) {
        __async_stream[i].get();
    }
}



template <int threadDim_x, int threadDim_y>
void decx::sgemm_Dblock_FH_FL_LW16(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
    const int wB_left, const size_t& fake_wB)
{
    std::future<void> __async_stream[threadDim_x * threadDim_y];

    int sum = 0;

    for (int i = 0; i < threadDim_x; ++i) {
        for (int j = 0; j < threadDim_y - 1; ++j) {
            __async_stream[sum] = decx::thread_pool.register_task(
                decx::_ST_sgemm_Dblock_FH_FL_W16,
                A + i * (size_t)proc_dim->y * (size_t)glo_dim->x,
                B + j * (size_t)fake_wB,
                C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
                glo_dim->x, glo_dim->z, proc_dim->x, proc_dim->y);
            ++sum;
        }
    }

    for (int i = 0; i < threadDim_x; ++i) {
        __async_stream[sum] = decx::thread_pool.register_task(
            decx::_ST_sgemm_Dblock_FH_FL_W16,
            A + i * (size_t)proc_dim->y * (size_t)glo_dim->x,
            B + (threadDim_y - 1) * (size_t)fake_wB,
            C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + (threadDim_y - 1) * (size_t)proc_dim->y,
            glo_dim->x, glo_dim->z, proc_dim->x, wB_left);
        ++sum;
    }

    for (int i = 0; i < threadDim_x * threadDim_y; ++i) {
        __async_stream[i].get();
    }
}



template <int threadDim_x, int threadDim_y>
void decx::sgemm_Dblock_FH_LL_LW16(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
    const int wB_left, const size_t& fake_wB)
{
    std::future<void> __async_stream[threadDim_x * threadDim_y];
    int sum = 0;
    for (int i = 0; i < threadDim_x; ++i) {
        for (int j = 0; j < threadDim_y - 1; ++j) {
            __async_stream[sum] = decx::thread_pool.register_task(
                    decx::_ST_sgemm_Dblock_FH_FL_W16,
                    A + i * (size_t)proc_dim->y * (size_t)glo_dim->x,
                    B + j * (size_t)fake_wB,
                    C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
                    glo_dim->x, glo_dim->z, proc_dim->x, proc_dim->y);
            ++sum;
        }
    }

    for (int i = 0; i < threadDim_x; ++i) {
        __async_stream[sum] = decx::thread_pool.register_task(
            decx::_ST_sgemm_Dblock_FH_LL_W16,
            A + i * (size_t)proc_dim->y * (size_t)glo_dim->x,
            B + (threadDim_y - 1) * (size_t)fake_wB,
            C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + (threadDim_y - 1) * (size_t)proc_dim->y,
            glo_dim->x, glo_dim->z, proc_dim->x, wB_left);
        ++sum;
    }

    for (int i = 0; i < threadDim_x * threadDim_y; ++i) {
        __async_stream[i].get();
    }
}



template <int threadDim_x, int threadDim_y>
void decx::sgemm_Dblock_FH_FL_LW8(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
    const int wB_left, const size_t& fake_wB)
{
    std::future<void>* __async_stream = new std::future<void>[threadDim_x * threadDim_y];
    int sum = 0;
    for (int i = 0; i < threadDim_x; ++i) {
        for (int j = 0; j < threadDim_y - 1; ++j) {
            __async_stream[sum] = decx::thread_pool.register_task(
                decx::_ST_sgemm_Dblock_FH_FL_W16,
                A + i * (size_t)proc_dim->y * (size_t)glo_dim->x,
                B + j * (size_t)fake_wB,
                C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
                glo_dim->x, glo_dim->z, proc_dim->x, proc_dim->y);
            ++sum;
        }
    }

    for (int i = 0; i < threadDim_x; ++i) {
        __async_stream[sum] = decx::thread_pool.register_task(
            decx::_ST_sgemm_Dblock_FH_FL_W8,
            A + i * (size_t)proc_dim->y * (size_t)glo_dim->x,
            B + (threadDim_y - 1) * (size_t)fake_wB,
            C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + (threadDim_y - 1) * (size_t)proc_dim->y,
            glo_dim->x, glo_dim->z, proc_dim->x, wB_left);
        ++sum;
    }

    for (int i = 0; i < threadDim_x * threadDim_y; ++i) {
        __async_stream[i].get();
    }
}



template <int threadDim_x, int threadDim_y>
static void decx::sgemm_Dblock_FH_LL_LW8(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
    const int wB_left, const size_t& fake_wB)
{
    std::future<void>* __async_stream = new std::future<void>[threadDim_x * threadDim_y];
    int sum = 0;
    for (int i = 0; i < threadDim_x; ++i) {
        for (int j = 0; j < threadDim_y - 1; ++j) {
            __async_stream[sum] = decx::thread_pool.register_task(
                decx::_ST_sgemm_Dblock_FH_LL_W16,
                A + i * (size_t)proc_dim->y * (size_t)glo_dim->x,
                B + j * (size_t)fake_wB,
                C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
                glo_dim->x, glo_dim->z, proc_dim->x, proc_dim->y);
            ++sum;
        }
    }

    for (int i = 0; i < threadDim_x; ++i) {
        __async_stream[sum] = decx::thread_pool.register_task(
            decx::_ST_sgemm_Dblock_FH_LL_W8,
            A + i * (size_t)proc_dim->y * (size_t)glo_dim->x,
            B + (threadDim_y - 1) * (size_t)fake_wB,
            C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + (threadDim_y - 1) * (size_t)proc_dim->y,
            glo_dim->x, glo_dim->z, proc_dim->x, wB_left);
        ++sum;
    }

    for (int i = 0; i < threadDim_x * threadDim_y; ++i) {
        __async_stream[i].get();
    }
}



template <int threadDim_x, int threadDim_y>
static void decx::sgemm_Dblock_LH_FL_FW(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim, 
    const int hA_left, const size_t& fake_wB)
{
    std::future<void> __async_stream[threadDim_x * threadDim_y];
    int sum = 0;

    for (int i = 0; i < threadDim_x - 1; ++i) {
        for (int j = 0; j < threadDim_y; ++j) {
            __async_stream[sum] = decx::thread_pool.register_task(
                decx::_ST_sgemm_Dblock_FH_FL_W16,
                A + i * (size_t)proc_dim->x * (size_t)glo_dim->x,
                B + j * (size_t)fake_wB,
                C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
                glo_dim->x, glo_dim->z, proc_dim->x, proc_dim->y);
            ++sum;
        }
    }
    for (int j = 0; j < threadDim_y; ++j) {
        __async_stream[sum] = decx::thread_pool.register_task(
            decx::_ST_sgemm_Dblock_LH_FL_W16,
            A + (threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->x,
            B + j * (size_t)fake_wB,
            C + ((threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
            glo_dim->x, glo_dim->z, hA_left, proc_dim->y);
        ++sum;
    }
    
    for (int i = 0; i < threadDim_x * threadDim_y; ++i) {
        __async_stream[i].get();
    }
}



template <int threadDim_x, int threadDim_y>
void decx::sgemm_Dblock_LH_FL_LW16(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
    const int wB_left, const int hA_left, const size_t& fake_wB)
{
    std::future<void> __async_stream[threadDim_x * threadDim_y];

    int sum = 0;

    for (int i = 0; i < threadDim_x - 1; ++i) {
        for (int j = 0; j < threadDim_y - 1; ++j) {
            __async_stream[sum] = decx::thread_pool.register_task(
                decx::_ST_sgemm_Dblock_FH_FL_W16,
                A + i * (size_t)proc_dim->x * (size_t)glo_dim->x,
                B + j * (size_t)fake_wB,
                C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
                glo_dim->x, glo_dim->z, proc_dim->x, proc_dim->y);
            ++sum;
        }
    }
    for (int j = 0; j < threadDim_y - 1; ++j) {
        __async_stream[sum] = decx::thread_pool.register_task(
            decx::_ST_sgemm_Dblock_LH_FL_W16,
            A + (threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->x,
            B + j * (size_t)fake_wB,
            C + ((threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
            glo_dim->x, glo_dim->z, hA_left, proc_dim->y);
        ++sum;
    }
    // left weight_B
    for (int i = 0; i < threadDim_x - 1; ++i) {
        __async_stream[sum] = decx::thread_pool.register_task(
            decx::_ST_sgemm_Dblock_FH_FL_W16,
            A + i * (size_t)proc_dim->x * (size_t)glo_dim->x,
            B + (threadDim_y - 1) * (size_t)fake_wB,
            C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + (threadDim_y - 1) * (size_t)proc_dim->y,
            glo_dim->x, glo_dim->z, proc_dim->x, wB_left);
        ++sum;
    }
    __async_stream[sum] = decx::thread_pool.register_task(
        decx::_ST_sgemm_Dblock_LH_FL_W16,
        A + (threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->x,
        B + (threadDim_y - 1) * (size_t)fake_wB,
        C + ((threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->z) + (threadDim_y - 1) * (size_t)proc_dim->y,
        glo_dim->x, glo_dim->z, hA_left, wB_left);

    for (int i = 0; i < threadDim_x * threadDim_y; ++i) {
        __async_stream[i].get();
    }
}




template <int threadDim_x, int threadDim_y>
void decx::sgemm_Dblock_LH_LL_FW(float* A, float* B, float* C, const int4* glo_dim, 
    const int2* proc_dim, const int hA_left, const size_t& fake_wB)
{
    std::future<void> __async_stream[threadDim_x * threadDim_y];
    int sum = 0;

    for (int i = 0; i < threadDim_x - 1; ++i) {
        for (int j = 0; j < threadDim_y; ++j) {
            __async_stream[sum] = decx::thread_pool.register_task(
                decx::_ST_sgemm_Dblock_FH_LL_W16,
                A + i * (size_t)proc_dim->x * (size_t)glo_dim->x,
                B + j * (size_t)fake_wB,
                C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
                glo_dim->x, glo_dim->z, proc_dim->x, proc_dim->y);
            ++sum;
        }
    }
    for (int j = 0; j < threadDim_y; ++j) {
        __async_stream[sum] = decx::thread_pool.register_task(
            decx::_ST_sgemm_Dblock_LH_LL_W16,
            A + (threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->x,
            B + j * (size_t)fake_wB,
            C + ((threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
            glo_dim->x, glo_dim->z, hA_left, proc_dim->y);
        ++sum;
    }

    for (int i = 0; i < threadDim_x * threadDim_y; ++i) {
        __async_stream[i].get();
    }
}



template <int threadDim_x, int threadDim_y>
void decx::sgemm_Dblock_LH_LL_LW16(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
    const int wB_left, const int hA_left, const size_t& fake_wB)
{
    std::future<void> __async_stream[threadDim_x * threadDim_y];

    int sum = 0;

    for (int i = 0; i < threadDim_x - 1; ++i) {
        for (int j = 0; j < threadDim_y - 1; ++j) {
            __async_stream[sum] = decx::thread_pool.register_task(
                decx::_ST_sgemm_Dblock_FH_LL_W16,
                A + i * (size_t)proc_dim->x * (size_t)glo_dim->x,
                B + j * (size_t)fake_wB,
                C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
                glo_dim->x, glo_dim->z, proc_dim->x, proc_dim->y);
            ++sum;
        }
    }
    for (int j = 0; j < threadDim_y - 1; ++j) {
        __async_stream[sum] = decx::thread_pool.register_task(
            decx::_ST_sgemm_Dblock_LH_LL_W16,
            A + (threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->x,
            B + j * (size_t)fake_wB,
            C + ((threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
            glo_dim->x, glo_dim->z, hA_left, proc_dim->y);
        ++sum;
    }
    // left weight_B
    for (int i = 0; i < threadDim_x - 1; ++i) {
        __async_stream[sum] = decx::thread_pool.register_task(
            decx::_ST_sgemm_Dblock_FH_LL_W16,
            A + i * (size_t)proc_dim->x * (size_t)glo_dim->x,
            B + (threadDim_y - 1) * (size_t)fake_wB,
            C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + (threadDim_y - 1) * (size_t)proc_dim->y,
            glo_dim->x, glo_dim->z, proc_dim->x, wB_left);
        ++sum;
    }
    __async_stream[sum] = decx::thread_pool.register_task(
        decx::_ST_sgemm_Dblock_LH_LL_W16,
        A + (threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->x,
        B + (threadDim_y - 1) * (size_t)fake_wB,
        C + ((threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->z) + (threadDim_y - 1) * (size_t)proc_dim->y,
        glo_dim->x, glo_dim->z, hA_left, wB_left);

    for (int i = 0; i < threadDim_x * threadDim_y; ++i) {
        __async_stream[i].get();
    }
}




template <int threadDim_x, int threadDim_y>
void decx::sgemm_Dblock_LH_FL_LW8(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
    const int wB_left, const int hA_left, const size_t& fake_wB)
{
    std::future<void>* __async_stream = new std::future<void>[threadDim_x * threadDim_y];
    int sum = 0;
    for (int i = 0; i < threadDim_x - 1; ++i) {
        for (int j = 0; j < threadDim_y - 1; ++j) {
            __async_stream[sum] = decx::thread_pool.register_task(
                decx::_ST_sgemm_Dblock_FH_FL_W16,
                A + i * (size_t)proc_dim->x * (size_t)glo_dim->x,
                B + j * (size_t)fake_wB,
                C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
                glo_dim->x, glo_dim->z, proc_dim->x, proc_dim->y);
            ++sum;
        }
    }
    for (int j = 0; j < threadDim_y - 1; ++j) {
        __async_stream[sum] = decx::thread_pool.register_task(
            decx::_ST_sgemm_Dblock_LH_FL_W16,
            A + (threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->x,
            B + j * (size_t)fake_wB,
            C + ((threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
            glo_dim->x, glo_dim->z, hA_left, proc_dim->y);
        ++sum;
    }
    // left wieght_B
    for (int i = 0; i < threadDim_x - 1; ++i) {
        __async_stream[sum] = decx::thread_pool.register_task(
            decx::_ST_sgemm_Dblock_FH_FL_W8,
            A + i * (size_t)proc_dim->x * (size_t)glo_dim->x,
            B + (threadDim_y - 1) * (size_t)fake_wB,
            C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + (threadDim_y - 1) * (size_t)proc_dim->y,
            glo_dim->x, glo_dim->z, proc_dim->x, wB_left);
        ++sum;
    }
    __async_stream[sum] = decx::thread_pool.register_task(
        decx::_ST_sgemm_Dblock_LH_FL_W8,
        A + (threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->x,
        B + (threadDim_y - 1) * (size_t)fake_wB,
        C + ((threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->z) + (threadDim_y - 1) * (size_t)proc_dim->y,
        glo_dim->x, glo_dim->z, hA_left, wB_left);

    for (int i = 0; i < threadDim_x * threadDim_y; ++i) {
        __async_stream[i].get();
    }
}




template <int threadDim_x, int threadDim_y>
void decx::sgemm_Dblock_LH_LL_LW8(float* A, float* B, float* C, const int4* glo_dim, const int2* proc_dim,
    const int wB_left, const int hA_left, const size_t& fake_wB)
{
    std::future<void>* __async_stream = new std::future<void>[threadDim_x * threadDim_y];
    int sum = 0;
    for (int i = 0; i < threadDim_x - 1; ++i) {
        for (int j = 0; j < threadDim_y - 1; ++j) {
            __async_stream[sum] = decx::thread_pool.register_task(
                decx::_ST_sgemm_Dblock_FH_LL_W16,
                A + i * (size_t)proc_dim->x * (size_t)glo_dim->x,
                B + j * (size_t)fake_wB,
                C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
                glo_dim->x, glo_dim->z, proc_dim->x, proc_dim->y);
            ++sum;
        }
    }
    for (int j = 0; j < threadDim_y - 1; ++j) {
        __async_stream[sum] = decx::thread_pool.register_task(
            decx::_ST_sgemm_Dblock_LH_LL_W16,
            A + (threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->x,
            B + j * (size_t)fake_wB,
            C + ((threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->z) + j * (size_t)proc_dim->y,
            glo_dim->x, glo_dim->z, hA_left, proc_dim->y);
        ++sum;
    }
    // left wieght_B
    for (int i = 0; i < threadDim_x - 1; ++i) {
        __async_stream[sum] = decx::thread_pool.register_task(
            decx::_ST_sgemm_Dblock_FH_LL_W8,
            A + i * (size_t)proc_dim->x * (size_t)glo_dim->x,
            B + (threadDim_y - 1) * (size_t)fake_wB,
            C + (i * (size_t)proc_dim->x * (size_t)glo_dim->z) + (threadDim_y - 1) * (size_t)proc_dim->y,
            glo_dim->x, glo_dim->z, proc_dim->x, wB_left);
        ++sum;
    }
    __async_stream[sum] = decx::thread_pool.register_task(
        decx::_ST_sgemm_Dblock_LH_LL_W8,
        A + (threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->x,
        B + (threadDim_y - 1) * (size_t)fake_wB,
        C + ((threadDim_x - 1) * (size_t)proc_dim->x * (size_t)glo_dim->z) + (threadDim_y - 1) * (size_t)proc_dim->y,
        glo_dim->x, glo_dim->z, hA_left, wB_left);

    for (int i = 0; i < threadDim_x * threadDim_y; ++i) {
        __async_stream[i].get();
    }
}



// ----------------------------------------------------------------------------------------------------------


template <int threadDim_x, int threadDim_y>
static void decx::sgemm_caller_FH_W16(float* A, float* B, float* C, const int4* glo_dim)
{
    int lane_num = glo_dim->z / 16;
    int2 proc_dim = make_int2(glo_dim->y / threadDim_x, (lane_num / threadDim_y) * 16);
    size_t fake_wB = (size_t)glo_dim->x * (size_t)proc_dim.y;

    if (glo_dim->x % sgemm_BL_Linear) {        // Left Linear
        if (lane_num % threadDim_y)
        {       // LW16
            decx::sgemm_Dblock_FH_LL_LW16<threadDim_x, threadDim_y>(A, B, C, glo_dim, &proc_dim,
                ((lane_num / threadDim_y) + (lane_num % threadDim_y)) * 16,
                fake_wB);
        }
        else {      // FW
            decx::sgemm_Dblock_FH_LL_FW<threadDim_x, threadDim_y>(
                A, B, C, glo_dim, &proc_dim, fake_wB);
        }
    }
    else {      // Full Linear
        if (lane_num % threadDim_y) {       // LW16
            decx::sgemm_Dblock_FH_FL_LW16<threadDim_x, threadDim_y>(
                A, B, C, glo_dim, &proc_dim, 
                ((lane_num / threadDim_y) + (lane_num % threadDim_y)) * 16, 
                fake_wB);
        }
        else {      // FW
            decx::sgemm_Dblock_FH_FL_FW<threadDim_x, threadDim_y>(
                A, B, C, glo_dim, &proc_dim, fake_wB);
        }
    }
}


template <int threadDim_x, int threadDim_y>
static void decx::sgemm_caller_LH_W16(float* A, float* B, float* C, const int4* glo_dim)
{
    int Wlane = glo_dim->z / 16;
    int Hlane = glo_dim->y / sgemm_BL_hA;

    int2 proc_dim = make_int2((Hlane / threadDim_x) * sgemm_BL_hA, (Wlane / threadDim_y) * 16);
    size_t fake_wB = (size_t)glo_dim->x * (size_t)proc_dim.y;
    int HLeft = ((Hlane / threadDim_x) + (Hlane % threadDim_x)) * sgemm_BL_hA + (glo_dim->y % sgemm_BL_hA);

    if (glo_dim->x % sgemm_BL_Linear) {        // Left Linear
        if (Wlane % threadDim_y){       // Left Width_B 16x
            decx::sgemm_Dblock_LH_LL_LW16<threadDim_x, threadDim_y>(A, B, C, glo_dim, &proc_dim,
                ((Wlane / threadDim_y) + (Wlane % threadDim_y)) * 16,
                HLeft, fake_wB);
        }
        else {      // Full Width_B
            decx::sgemm_Dblock_LH_LL_FW<threadDim_x, threadDim_y>(
                A, B, C, glo_dim, &proc_dim, HLeft, fake_wB);
        }
    }
    else {      // Full Linear
        if (Wlane % threadDim_y) {       // Left Width_B 16x
            decx::sgemm_Dblock_LH_FL_LW16<threadDim_x, threadDim_y>(
                A, B, C, glo_dim, &proc_dim,
                ((Wlane / threadDim_y) + (Wlane % threadDim_y)) * 16,
                HLeft, fake_wB);
        }
        else {      // Full Width_B
            decx::sgemm_Dblock_LH_FL_FW<threadDim_x, threadDim_y>(
                A, B, C, glo_dim, &proc_dim, HLeft, fake_wB);
        }
    }
}


template <int threadDim_x, int threadDim_y>
static void decx::sgemm_caller_FH_W8(float* A, float* B, float* C, const int4* glo_dim)
{
    int lane_num = (glo_dim->z - 8) / 16;
    int2 proc_dim = make_int2(glo_dim->y / threadDim_x, (lane_num / threadDim_y) * 16);
    size_t fake_wB = (size_t)glo_dim->x * (size_t)proc_dim.y;
    int _wB_left = ((lane_num / threadDim_y) + (lane_num % threadDim_y)) * 16 + 8;

    if (glo_dim->x % sgemm_BL_Linear) {        // LL
        decx::sgemm_Dblock_FH_LL_LW8<threadDim_x, threadDim_y>(A, B, C, glo_dim, &proc_dim, _wB_left, fake_wB);
    }
    else {      // FL
        decx::sgemm_Dblock_FH_FL_LW8<threadDim_x, threadDim_y>(A, B, C, glo_dim, &proc_dim, _wB_left, fake_wB);
    }
}


template <int threadDim_x, int threadDim_y>
static void decx::sgemm_caller_LH_W8(float* A, float* B, float* C, const int4* glo_dim)
{
    int lane_num = (glo_dim->z - 8) / 16;
    int Hlane = glo_dim->y / sgemm_BL_hA;

    int2 proc_dim = make_int2((Hlane / threadDim_x) * sgemm_BL_hA, (lane_num / threadDim_y) * 16);
    size_t fake_wB = (size_t)glo_dim->x * (size_t)proc_dim.y;
    int _wB_left = ((lane_num / threadDim_y) + (lane_num % threadDim_y)) * 16 + 8;

    int HLeft = ((Hlane / threadDim_x) + (Hlane % threadDim_x)) * sgemm_BL_hA + (glo_dim->y % sgemm_BL_hA);

    if (glo_dim->x % sgemm_BL_Linear) {        // Left Linear
        decx::sgemm_Dblock_LH_LL_LW8<threadDim_x, threadDim_y>(A, B, C, glo_dim, &proc_dim, _wB_left, HLeft, fake_wB);
    }
    else {      // Full Linear
        decx::sgemm_Dblock_LH_FL_LW8<threadDim_x, threadDim_y>(A, B, C, glo_dim, &proc_dim, _wB_left, HLeft, fake_wB);
    }
}


// -------------------------------------------------------------------------------------------


template <int threadDim_x, int threadDim_y>
static void decx::sgemm_caller(float* A, float* B, float* tmp_B, float* dst, const int4 *glo_dim)
{
    int2 sort_dims_info = make_int2(glo_dim->x, glo_dim->z);
    if (glo_dim->z % 16) {      // width_B = 16N + 8
        decx::sort_MatB_16_w8<threadDim_x * threadDim_y>(B, tmp_B, sort_dims_info);
        if (glo_dim->y % sgemm_BL_hA) {     // left height_A
            decx::sgemm_caller_LH_W8<threadDim_x, threadDim_y>(A, tmp_B, dst, glo_dim);
        }
        else {      // full height_A
            decx::sgemm_caller_FH_W8<threadDim_x, threadDim_y>(A, tmp_B, dst, glo_dim);
        }
    }
    else {      // width_B = 16N
        decx::sort_MatB_16<threadDim_x * threadDim_y>(B, tmp_B, sort_dims_info);
        if (glo_dim->y % sgemm_BL_hA) {     // left height_A
            decx::sgemm_caller_LH_W16<threadDim_x, threadDim_y>(A, tmp_B, dst, glo_dim);
        }
        else {      // full height_A
            decx::sgemm_caller_FH_W16<threadDim_x, threadDim_y>(A, tmp_B, dst, glo_dim);
        }
    }
}


#endif