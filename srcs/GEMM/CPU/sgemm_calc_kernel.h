/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _SGEMM_CALC_KERNEL_H_
#define _SGEMM_CALC_KERNEL_H_

#include "gemm_utils.h"


#define sgemm_BL_Linear 64      // 8次迭代
#define sgemm_BL_hA 32          // 32次迭代
#define sgemm_BL_wB 16          // 2次迭代
#define sgemm_BL_linear_iter 8  // = sgemm_BL_Linear / 4(length of __m128)



/* (__linear, wB, hA) */
#define _SGEMM_CALC_KERNEL_16(a_dex){    \
tp_b[0] = _mm256_load_ps(B + dex_B);       dex_B += 8;  \
tp_b[1] = _mm256_load_ps(B + dex_B);       dex_B += 8;  \
tp_c[0] = _mm256_fmadd_ps(_mm256_set1_ps(tp_a.m256_f32[a_dex]), tp_b[0], tp_c[0]);  \
tp_c[1] = _mm256_fmadd_ps(_mm256_set1_ps(tp_a.m256_f32[a_dex]), tp_b[1], tp_c[1]);  \
}


#define _SGEMM_CALC_KERNEL_8(a_dex){    \
tp_b = _mm256_load_ps(B + dex_B);          dex_B += 8;  \
tp_c = _mm256_fmadd_ps(_mm256_set1_ps(tp_a.m256_f32[a_dex]), tp_b, tp_c);   \
}



namespace decx
{
    /* (32, 32, 32)
    * dims_info.x -> wA (linear)
    * dims_info.y -> wB (8x satisfied) = wC
    */
    void _THREAD_FUNCTION_ _avx256_sgemm_block(float* A, float* B, float* C, const int __linear, const int global_wB);

    /**
     * @brief (32, 8, 32), It can automatically adapt to __linear
     * @param A Matrix A
     * @param B Matrix B(temp)
     * @param C Matrix C(dst)
     * @param dims_info
     * dims_info.x -> wA (linear)
     * dims_info.y -> wB (8x satisfied) = wC
     * dims_info.z -> fake_wB
     */
    void _THREAD_FUNCTION_ _avx256_sgemm_block_w8(float* A, float* B, float* C, const int __linear, const int global_wB);

    /**
     * @brief (32, 8, 32), It can automatically adapt to __linear
     * @param A Matrix A
     * @param B Matrix B(temp)
     * @param C Matrix C(dst)
     * @param dims_info
     * dims_info.x -> wA (linear)
     * dims_info.y -> wB (8x satisfied) = wC
     * dims_info.z -> fake_wB
     */
    void _THREAD_FUNCTION_ _avx256_sgemm_block_flexibleH_w8(float* A, float* B, float* C, const int __linear, const int global_wB, const int& _HLeft);

    /* (_Lleft, 32, 32)
    * dims_info.x -> wA (linear)
    * dims_info.y -> wB
    * dims_info.z -> fake_wB
    * Linear_Num 4-times on linear region
    */
    void _THREAD_FUNCTION_ _avx256_sgemm_block_Lleft(float* A, float* B, float* C, const int __linear, const int global_wB, const int _Lleft);

    /**
     * @brief (32, 8, 32), It can automatically adapt to __linear
     * @param A Matrix A
     * @param B Matrix B(temp)
     * @param C Matrix C(dst)
     * @param dims_info
     * dims_info.x -> wA (linear)
     * dims_info.y -> wB (8x satisfied) = wC
     * dims_info.z -> fake_wB
     */
    void _THREAD_FUNCTION_ _avx256_sgemm_block_Lleft_w8(float* A, float* B, float* C, const int __linear, const int global_wB, const int _Lleft);

    /**
     * @brief (32, 8, 32), It can automatically adapt to __linear
     * @param A Matrix A
     * @param B Matrix B(temp)
     * @param C Matrix C(dst)
     * @param dims_info
     * dims_info.x -> wA (linear)
     * dims_info.y -> wB (8x satisfied) = wC
     * dims_info.z -> fake_wB
     */
    void _THREAD_FUNCTION_ _avx256_sgemm_block_Lleft_flexibleH_w8(float* A, float* B, float* C,
        const int __linear, const int global_wB, const int& _Lleft, const int& _HLeft);

    /**
     * @brief (32, 8, 32), It can automatically adapt to __linear
     * @param A Matrix A
     * @param B Matrix B(temp)
     * @param C Matrix C(dst)
     * @param dims_info
     * dims_info.x -> wA (linear)
     * dims_info.y -> wB (8x satisfied) = wC
     * dims_info.z -> fake_wB
     */
    void _THREAD_FUNCTION_ _avx256_sgemm_block_Lleft_flexibleH_w8(float* A, float* B, float* C,
        const int __linear, const int global_wB, const int& _Lleft, const int& _HLeft);

    /* (32, 32, 32)
    * dims_info.x -> wA (linear)
    * dims_info.y -> wB (8x satisfied) = wC
    * dims_info.z -> fake_wB
    */
    void _THREAD_FUNCTION_ _avx256_sgemm_block_first(float* A, float* B, float* C, const int __linear, const int global_wB);

    /**
     * @brief (32, 8, 32), It can automatically adapt to __linear
     * @param A Matrix A
     * @param B Matrix B(temp)
     * @param C Matrix C(dst)
     * @param dims_info
     * dims_info.x -> wA (linear)
     * dims_info.y -> wB (8x satisfied) = wC
     * dims_info.z -> fake_wB
     */
    void _THREAD_FUNCTION_ _avx256_sgemm_block_first_w8(float* A, float* B, float* C, const int __linear, const int global_wB);

    /**
     * @brief (32, 8, 32), It can automatically adapt to __linear
     * @param A Matrix A
     * @param B Matrix B(temp)
     * @param C Matrix C(dst)
     * @param dims_info
     * dims_info.x -> wA (linear)
     * dims_info.y -> wB (8x satisfied) = wC
     * dims_info.z -> fake_wB
     */
    void _THREAD_FUNCTION_ _avx256_sgemm_block_first_flexibleH_w8(float* A, float* B, float* C,
        const int __linear, const int global_wB, const int& _HLeft);

    /* (32, 32, 32)
    * dims_info.x -> wA (linear)
    * dims_info.y -> wB (8x satisfied) = wC
    */
    void _THREAD_FUNCTION_ _avx256_sgemm_block_flexibleH(float* A, float* B, float* C,
        const int __linear, const int global_wB, const int _HLeft);

    /**
     * @brief (32, 8, 32), It can automatically adapt to __linear
     * @param A Matrix A
     * @param B Matrix B(temp)
     * @param C Matrix C(dst)
     * @param dims_info
     * dims_info.x -> wA (linear)
     * dims_info.y -> wB (8x satisfied) = wC
     */
    void _THREAD_FUNCTION_ _avx256_sgemm_block_w8_flexibleH(float* A, float* B, float* C,
        const int __linear, const int global_wB, const int _HLeft);

    /* (_Lleft, 32, 32)
    * dims_info.x -> wA (linear)
    * dims_info.y -> wB
    * dims_info.z -> fake_wB
    * Linear_Num 4-times on linear region
    */
    void _THREAD_FUNCTION_ _avx256_sgemm_block_Lleft_flexibleH(float* A, float* B, float* C,
        const int __linear, const int global_wB, const int _Lleft, const int _HLeft);

    /* (32, 32, 32)
    * dims_info.x -> wA (linear)
    * dims_info.y -> wB (8x satisfied) = wC
    * dims_info.z -> fake_wB
    */
    void _THREAD_FUNCTION_ _avx256_sgemm_block_flexibleH_first(float* A, float* B, float* C,
        const int __linear, const int global_wB, const int _HLeft);
}


/* (__linear, wB, hA) */


void _THREAD_FUNCTION_ decx::_avx256_sgemm_block(float* A, float* B, float* C, const int __linear, const int global_wB)
{
    __m256 tp_a, tp_b[2], tp_c[2];

    size_t dex_A = 0, dex_B = 0,
        dex_C = 0;

    // if I am not using 'volatile', the result will be wrong because of the optimization
    // And if I am not putting it into register, the memory access will be extremely ineffecient
    volatile register int
        i = 0,
        j = 0,
        z = 0;

    for (i = 0; i < sgemm_BL_hA; ++i) {
        dex_B = 0;

        tp_c[0] = _mm256_load_ps(C + dex_C);        dex_C += 8;
        tp_c[1] = _mm256_load_ps(C + dex_C);

        for (z = 0; z < sgemm_BL_linear_iter; ++z) {
            tp_a = _mm256_load_ps(A + dex_A);
            dex_A += 8;

            _SGEMM_CALC_KERNEL_16(0)
            _SGEMM_CALC_KERNEL_16(1)
            _SGEMM_CALC_KERNEL_16(2)
            _SGEMM_CALC_KERNEL_16(3)
            _SGEMM_CALC_KERNEL_16(4)
            _SGEMM_CALC_KERNEL_16(5)
            _SGEMM_CALC_KERNEL_16(6)
            _SGEMM_CALC_KERNEL_16(7)

        }
        dex_C -= 8;

        _mm256_store_ps(C + dex_C, tp_c[0]);        dex_C += 8;
        _mm256_store_ps(C + dex_C, tp_c[1]);        dex_C += 8;

        dex_C += (size_t)global_wB - sgemm_BL_wB;
        dex_A += (size_t)__linear - sgemm_BL_Linear;
    }
}



void _THREAD_FUNCTION_ decx::_avx256_sgemm_block_w8(float* A, float* B, float* C, const int __linear, const int global_wB)
{
    __m256 tp_a;
    __m256 tp_b,
        tp_c;

    size_t dex_A = 0, dex_B = 0,
        tmp_dex_A,
        dex_C = 0;

    // if I am not using 'volatile', the result will be wrong because of the optimization
    // And if I am not putting it into register, the memory access will be extremely ineffecient
    volatile register uchar
        i = 0,
        j = 0,
        z = 0;

    for (i = 0; i < sgemm_BL_hA; ++i) {
        dex_B = 0;
        tp_c = _mm256_load_ps(C + dex_C);

        for (z = 0; z < sgemm_BL_linear_iter; ++z) {
            tp_a = _mm256_load_ps(A + dex_A);
            dex_A += 8;

            _SGEMM_CALC_KERNEL_8(0)
            _SGEMM_CALC_KERNEL_8(1)
            _SGEMM_CALC_KERNEL_8(2)
            _SGEMM_CALC_KERNEL_8(3)
            _SGEMM_CALC_KERNEL_8(4)
            _SGEMM_CALC_KERNEL_8(5)
            _SGEMM_CALC_KERNEL_8(6)
            _SGEMM_CALC_KERNEL_8(7)
        }

        _mm256_store_ps(C + dex_C, tp_c);
        dex_C += (size_t)global_wB;
        dex_A += (size_t)__linear - sgemm_BL_Linear;
    }
}



void _THREAD_FUNCTION_ decx::_avx256_sgemm_block_flexibleH_w8(float* A, float* B, float* C, const int __linear, const int global_wB, const int& _HLeft)
{
    __m256 tp_a;
    __m256 tp_b,
        tp_c;

    size_t dex_A = 0, dex_B = 0,
        tmp_dex_A,
        dex_C = 0;

    // if I am not using 'volatile', the result will be wrong because of the optimization
    // And if I am not putting it into register, the memory access will be extremely ineffecient
    volatile register uchar
        i = 0,
        j = 0,
        z = 0;

    for (i = 0; i < _HLeft; ++i) {
        dex_B = 0;
        tp_c = _mm256_load_ps(C + dex_C);

        for (z = 0; z < sgemm_BL_linear_iter; ++z) {
            tp_a = _mm256_load_ps(A + dex_A);
            dex_A += 8;

            _SGEMM_CALC_KERNEL_8(0)
            _SGEMM_CALC_KERNEL_8(1)
            _SGEMM_CALC_KERNEL_8(2)
            _SGEMM_CALC_KERNEL_8(3)
            _SGEMM_CALC_KERNEL_8(4)
            _SGEMM_CALC_KERNEL_8(5)
            _SGEMM_CALC_KERNEL_8(6)
            _SGEMM_CALC_KERNEL_8(7)
        }

        _mm256_store_ps(C + dex_C, tp_c);
        dex_C += (size_t)global_wB;
        dex_A += (size_t)__linear - sgemm_BL_Linear;
    }
}



void _THREAD_FUNCTION_ decx::_avx256_sgemm_block_Lleft(float* A, float* B, float* C, const int __linear, const int global_wB, const int _Lleft)
{
    __m256 tp_a;
    __m256 tp_b[2],
        tp_c[2];

    size_t dex_A = 0, dex_B = 0,
        dex_A_gap = (size_t)__linear - (size_t)_Lleft,
        dex_C = 0;

    // if I am not using 'volatile', the result will be wrong because of the optimization
    // And if I am not putting it into register, the memory access will be extremely ineffecient
    volatile register int
        i = 0,
        j = 0,
        z = 0;

    for (i = 0; i < sgemm_BL_hA; ++i) {
        dex_B = 0;

        tp_c[0] = _mm256_load_ps(C + dex_C);        dex_C += 8;
        tp_c[1] = _mm256_load_ps(C + dex_C);

        for (z = 0; z < _Lleft / 8; ++z) {
            tp_a = _mm256_load_ps(A + dex_A);
            dex_A += 8;

            _SGEMM_CALC_KERNEL_16(0)
            _SGEMM_CALC_KERNEL_16(1)
            _SGEMM_CALC_KERNEL_16(2)
            _SGEMM_CALC_KERNEL_16(3)
            _SGEMM_CALC_KERNEL_16(4)
            _SGEMM_CALC_KERNEL_16(5)
            _SGEMM_CALC_KERNEL_16(6)
            _SGEMM_CALC_KERNEL_16(7)
        }

        dex_C -= 8;

        _mm256_store_ps(C + dex_C, tp_c[0]);        dex_C += 8;
        _mm256_store_ps(C + dex_C, tp_c[1]);        dex_C += 8;

        dex_C += (size_t)global_wB - sgemm_BL_wB;
        dex_A += dex_A_gap;
    }
}



void _THREAD_FUNCTION_ decx::_avx256_sgemm_block_Lleft_w8(float* A, float* B, float* C, const int __linear, const int global_wB, const int _Lleft)
{
    __m256 tp_a;
    __m256 tp_b,
        tp_c;

    size_t dex_A = 0, dex_B = 0,
        tmp_dex_A,
        dex_C = 0;

    // if I am not using 'volatile', the result will be wrong because of the optimization
    // And if I am not putting it into register, the memory access will be extremely ineffecient
    volatile register uchar
        i = 0,
        j = 0,
        z = 0;

    for (i = 0; i < sgemm_BL_hA; ++i) {
        dex_B = 0;
        tp_c = _mm256_load_ps(C + dex_C);

        for (z = 0; z < _Lleft / 8; ++z) {
            tp_a = _mm256_load_ps(A + dex_A);
            dex_A += 8;

            _SGEMM_CALC_KERNEL_8(0)
            _SGEMM_CALC_KERNEL_8(1)
            _SGEMM_CALC_KERNEL_8(2)
            _SGEMM_CALC_KERNEL_8(3)
            _SGEMM_CALC_KERNEL_8(4)
            _SGEMM_CALC_KERNEL_8(5)
            _SGEMM_CALC_KERNEL_8(6)
            _SGEMM_CALC_KERNEL_8(7)
        }

        _mm256_store_ps(C + dex_C, tp_c);
        dex_C += (size_t)global_wB;
        dex_A += (size_t)__linear - (size_t)_Lleft;
    }
}



void _THREAD_FUNCTION_ decx::_avx256_sgemm_block_Lleft_flexibleH_w8(float* A, float* B, float* C,
    const int __linear, const int global_wB, const int& _Lleft, const int& _HLeft)
{
    __m256 tp_a;
    __m256 tp_b,
        tp_c;

    size_t dex_A = 0, dex_B = 0,
        tmp_dex_A,
        dex_C = 0;

    // if I am not using 'volatile', the result will be wrong because of the optimization
    // And if I am not putting it into register, the memory access will be extremely ineffecient
    volatile register uchar
        i = 0,
        j = 0,
        z = 0;

    for (i = 0; i < _HLeft; ++i) {
        dex_B = 0;
        tp_c = _mm256_load_ps(C + dex_C);

        for (z = 0; z < _Lleft / 8; ++z) {
            tp_a = _mm256_load_ps(A + dex_A);
            dex_A += 8;

            _SGEMM_CALC_KERNEL_8(0)
            _SGEMM_CALC_KERNEL_8(1)
            _SGEMM_CALC_KERNEL_8(2)
            _SGEMM_CALC_KERNEL_8(3)
            _SGEMM_CALC_KERNEL_8(4)
            _SGEMM_CALC_KERNEL_8(5)
            _SGEMM_CALC_KERNEL_8(6)
            _SGEMM_CALC_KERNEL_8(7)
        }

        _mm256_store_ps(C + dex_C, tp_c);
        dex_C += (size_t)global_wB;
        dex_A += (size_t)__linear - (size_t)_Lleft;
    }
}



void _THREAD_FUNCTION_ decx::_avx256_sgemm_block_first(float* A, float* B, float* C, const int __linear, const int global_wB)
{
    __m256 tp_a;
    __m256 tp_b[2],
        tp_c[2];

    size_t dex_A = 0, dex_B = 0,
        dex_C = 0;

    // if I am not using 'volatile', the result will be wrong because of the optimization
    // And if I am not putting it into register, the memory access will be extremely ineffecient
    volatile register int
        i = 0,
        j = 0,
        z = 0;

    for (i = 0; i < sgemm_BL_hA; ++i) {
        dex_B = 0;
        tp_c[0] = _mm256_set1_ps(0.f);
        tp_c[1] = _mm256_set1_ps(0.f);

        for (z = 0; z < sgemm_BL_linear_iter; ++z) {
            tp_a = _mm256_load_ps(A + dex_A);
            dex_A += 8;

            _SGEMM_CALC_KERNEL_16(0)
            _SGEMM_CALC_KERNEL_16(1)
            _SGEMM_CALC_KERNEL_16(2)
            _SGEMM_CALC_KERNEL_16(3)
            _SGEMM_CALC_KERNEL_16(4)
            _SGEMM_CALC_KERNEL_16(5)
            _SGEMM_CALC_KERNEL_16(6)
            _SGEMM_CALC_KERNEL_16(7)
        }

        _mm256_store_ps(C + dex_C, tp_c[0]);        dex_C += 8;
        _mm256_store_ps(C + dex_C, tp_c[1]);        dex_C += 8;

        dex_C += (size_t)global_wB - sgemm_BL_wB;
        dex_A += (size_t)__linear - sgemm_BL_Linear;
    }
}



void _THREAD_FUNCTION_ decx::_avx256_sgemm_block_first_w8(float* A, float* B, float* C, const int __linear, const int global_wB)
{
    __m256 tp_a;
    __m256 tp_b,
        tp_c;

    size_t dex_A = 0, dex_B = 0,
        tmp_dex_A,
        dex_C = 0;

    // if I am not using 'volatile', the result will be wrong because of the optimization
    // And if I am not putting it into register, the memory access will be extremely ineffecient
    volatile register uchar
        i = 0,
        j = 0,
        z = 0;

    for (i = 0; i < sgemm_BL_hA; ++i) {
        dex_B = 0;
        tp_c = _mm256_set1_ps(0.f);

        for (z = 0; z < sgemm_BL_linear_iter; ++z) {
            tp_a = _mm256_load_ps(A + dex_A);
            dex_A += 8;

            _SGEMM_CALC_KERNEL_8(0)
            _SGEMM_CALC_KERNEL_8(1)
            _SGEMM_CALC_KERNEL_8(2)
            _SGEMM_CALC_KERNEL_8(3)
            _SGEMM_CALC_KERNEL_8(4)
            _SGEMM_CALC_KERNEL_8(5)
            _SGEMM_CALC_KERNEL_8(6)
            _SGEMM_CALC_KERNEL_8(7)
        }

        _mm256_store_ps(C + dex_C, tp_c);
        dex_C += (size_t)global_wB;
        dex_A += (size_t)__linear - sgemm_BL_Linear;
    }
}



void _THREAD_FUNCTION_ decx::_avx256_sgemm_block_first_flexibleH_w8(float* A, float* B, float* C,
    const int __linear, const int global_wB, const int& _HLeft)
{
    __m256 tp_a;
    __m256 tp_b,
        tp_c;

    size_t dex_A = 0, dex_B = 0,
        tmp_dex_A,
        dex_C = 0;

    // if I am not using 'volatile', the result will be wrong because of the optimization
    // And if I am not putting it into register, the memory access will be extremely ineffecient
    volatile register uchar
        i = 0,
        j = 0,
        z = 0;

    for (i = 0; i < _HLeft; ++i) {
        dex_B = 0;
        tp_c = _mm256_set1_ps(0.f);

        for (z = 0; z < sgemm_BL_linear_iter; ++z) {
            tp_a = _mm256_load_ps(A + dex_A);
            dex_A += 8;

            _SGEMM_CALC_KERNEL_8(0)
            _SGEMM_CALC_KERNEL_8(1)
            _SGEMM_CALC_KERNEL_8(2)
            _SGEMM_CALC_KERNEL_8(3)
            _SGEMM_CALC_KERNEL_8(4)
            _SGEMM_CALC_KERNEL_8(5)
            _SGEMM_CALC_KERNEL_8(6)
            _SGEMM_CALC_KERNEL_8(7)
        }

        _mm256_store_ps(C + dex_C, tp_c);
        dex_C += (size_t)global_wB;
        dex_A += (size_t)__linear - sgemm_BL_Linear;
    }
}



void _THREAD_FUNCTION_ decx::_avx256_sgemm_block_flexibleH(float* A, float* B, float* C,
    const int __linear, const int global_wB, const int _HLeft)
{
    __m256 tp_a;
    __m256 tp_b[2],
        tp_c[2];

    size_t dex_A = 0, dex_B = 0,
        dex_C = 0;

    // if I am not using 'volatile', the result will be wrong because of the optimization
    // And if I am not putting it into register, the memory access will be extremely ineffecient
    volatile register int
        i = 0,
        j = 0,
        z = 0;

    for (i = 0; i < _HLeft; ++i) {
        dex_B = 0;

        tp_c[0] = _mm256_load_ps(C + dex_C);        dex_C += 8;
        tp_c[1] = _mm256_load_ps(C + dex_C);

        for (z = 0; z < sgemm_BL_linear_iter; ++z) {
            tp_a = _mm256_load_ps(A + dex_A);
            dex_A += 8;

            _SGEMM_CALC_KERNEL_16(0)
            _SGEMM_CALC_KERNEL_16(1)
            _SGEMM_CALC_KERNEL_16(2)
            _SGEMM_CALC_KERNEL_16(3)
            _SGEMM_CALC_KERNEL_16(4)
            _SGEMM_CALC_KERNEL_16(5)
            _SGEMM_CALC_KERNEL_16(6)
            _SGEMM_CALC_KERNEL_16(7)
        }
        dex_C -= 8;

        _mm256_store_ps(C + dex_C, tp_c[0]);        dex_C += 8;
        _mm256_store_ps(C + dex_C, tp_c[1]);        dex_C += 8;

        dex_C += (size_t)global_wB - sgemm_BL_wB;
        dex_A += (size_t)__linear - sgemm_BL_Linear;
    }
}



void _THREAD_FUNCTION_ decx::_avx256_sgemm_block_w8_flexibleH(float* A, float* B, float* C,
    const int __linear, const int global_wB, const int _HLeft)
{
    __m256 tp_a;
    __m256 tp_b,
        tp_c;

    size_t dex_A = 0, dex_B = 0,
        tmp_dex_A,
        dex_C = 0;

    // if I am not using 'volatile', the result will be wrong because of the optimization
    // And if I am not putting it into register, the memory access will be extremely ineffecient
    volatile register uchar
        i = 0,
        j = 0,
        z = 0;

    for (i = 0; i < _HLeft; ++i) {
        dex_B = 0;
        tp_c = _mm256_load_ps(C + dex_C);

        for (z = 0; z < sgemm_BL_linear_iter; ++z) {
            tp_a = _mm256_load_ps(A + dex_A);
            dex_A += 8;

            _SGEMM_CALC_KERNEL_8(0)
            _SGEMM_CALC_KERNEL_8(1)
            _SGEMM_CALC_KERNEL_8(2)
            _SGEMM_CALC_KERNEL_8(3)
            _SGEMM_CALC_KERNEL_8(4)
            _SGEMM_CALC_KERNEL_8(5)
            _SGEMM_CALC_KERNEL_8(6)
            _SGEMM_CALC_KERNEL_8(7)
        }

        _mm256_store_ps(C + dex_C, tp_c);
        dex_C += (size_t)global_wB;
        dex_A += (size_t)__linear - sgemm_BL_Linear;
    }
}



void _THREAD_FUNCTION_ decx::_avx256_sgemm_block_Lleft_flexibleH(float* A, float* B, float* C,
    const int __linear, const int global_wB, const int _Lleft, const int _HLeft)
{
    __m256 tp_a;
    __m256 tp_b[2],
        tp_c[2];

    size_t dex_A = 0, dex_B = 0,
        dex_A_gap = (size_t)__linear - (size_t)_Lleft,
        dex_C = 0;

    // if I am not using 'volatile', the result will be wrong because of the optimization
    // And if I am not putting it into register, the memory access will be extremely ineffecient
    volatile register int
        i = 0,
        j = 0,
        z = 0;

    for (i = 0; i < _HLeft; ++i) {
        dex_B = 0;

        tp_c[0] = _mm256_load_ps(C + dex_C);        dex_C += 8;
        tp_c[1] = _mm256_load_ps(C + dex_C);

        for (z = 0; z < _Lleft / 8; ++z) {
            tp_a = _mm256_load_ps(A + dex_A);
            dex_A += 8;

            _SGEMM_CALC_KERNEL_16(0)
            _SGEMM_CALC_KERNEL_16(1)
            _SGEMM_CALC_KERNEL_16(2)
            _SGEMM_CALC_KERNEL_16(3)
            _SGEMM_CALC_KERNEL_16(4)
            _SGEMM_CALC_KERNEL_16(5)
            _SGEMM_CALC_KERNEL_16(6)
            _SGEMM_CALC_KERNEL_16(7)
        }

        dex_C -= 8;

        _mm256_store_ps(C + dex_C, tp_c[0]);        dex_C += 8;
        _mm256_store_ps(C + dex_C, tp_c[1]);        dex_C += 8;

        dex_C += (size_t)global_wB - sgemm_BL_wB;
        dex_A += dex_A_gap;
    }
}



void _THREAD_FUNCTION_ decx::_avx256_sgemm_block_flexibleH_first(float* A, float* B, float* C,
    const int __linear, const int global_wB, const int _HLeft)
{
    __m256 tp_a;
    __m256 tp_b[2],
        tp_c[2];

    size_t dex_A = 0, dex_B = 0,
        dex_C = 0;

    // if I am not using 'volatile', the result will be wrong because of the optimization
    // And if I am not putting it into register, the memory access will be extremely ineffecient
    volatile register int
        i = 0,
        j = 0,
        z = 0;

    for (i = 0; i < _HLeft; ++i) {
        dex_B = 0;
        tp_c[0] = _mm256_set1_ps(0.f);
        tp_c[1] = _mm256_set1_ps(0.f);

        for (z = 0; z < sgemm_BL_linear_iter; ++z) {
            tp_a = _mm256_load_ps(A + dex_A);
            dex_A += 8;

            _SGEMM_CALC_KERNEL_16(0)
            _SGEMM_CALC_KERNEL_16(1)
            _SGEMM_CALC_KERNEL_16(2)
            _SGEMM_CALC_KERNEL_16(3)
            _SGEMM_CALC_KERNEL_16(4)
            _SGEMM_CALC_KERNEL_16(5)
            _SGEMM_CALC_KERNEL_16(6)
            _SGEMM_CALC_KERNEL_16(7)
        }

        _mm256_store_ps(C + dex_C, tp_c[0]);        dex_C += 8;
        _mm256_store_ps(C + dex_C, tp_c[1]);        dex_C += 8;

        dex_C += (size_t)global_wB - sgemm_BL_wB;
        dex_A += (size_t)__linear - sgemm_BL_Linear;
    }
}


#endif