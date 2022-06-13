/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _GEMM_UTILS_H_
#define _GEMM_UTILS_H_

#include "../../core/basic.h"
#include "../../core/thread_management/thread_pool.h"



// ---------------------------- sort -------------------------------------------

namespace decx
{
    /*
    * dims_pkg.x : linear length
    * dims_pkg.y : wB (local)
    * dims_pkg.z : global_wB
    */
    void _THREAD_FUNCTION_ _avx256_sort_ST_MatB(float* srcB, float* dstB, const int4& dims_pkg);

    /**
    * At the very right side of Matrix B, there is only one __m256
    * dims_pkg.x : linear length
    * dims_pkg.y : wB (local)
    * dims_pkg.z : global_wB
    */
    void _THREAD_FUNCTION_ _avx256_sort_ST_MatB_w8(float* srcB, float* dstB, const int4& dims_pkg);

    /**
    * wB can be divided by 8 into integer
    * dims_pkg.x : linear length (hB) (wA)
    * dims_pkg.y : global_wB
    */
    template <int thread_num> void sort_MatB_16(float* srcB, float* dstB, const int2& dims_pkg);

    /**
    * wB can be divided by 8 into integer
    * dims_pkg.x : linear length (hB) (wA)
    * dims_pkg.y : global_wB
    */
    template <int thread_num> void sort_MatB_16_w8(float* srcB, float* dstB, const int2& dims_pkg);
}



void _THREAD_FUNCTION_ decx::_avx256_sort_ST_MatB(float* srcB, float* dstB, const int4& dims_pkg)
{
    size_t dex_src = 0, dex_dst = 0, tmp_dex_src = 0,
        true_B_gap = (size_t)dims_pkg.z - 8;

    for (int i = 0; i < dims_pkg.y; i += 16)
    {
        tmp_dex_src = dex_src;
        for (int _L = 0; _L < dims_pkg.x; ++_L)
        {
            _mm256_storeu_ps(dstB + dex_dst, _mm256_loadu_ps(srcB + tmp_dex_src));
            dex_dst += 8;
            tmp_dex_src += 8;
            _mm256_storeu_ps(dstB + dex_dst, _mm256_loadu_ps(srcB + tmp_dex_src));
            dex_dst += 8;
            tmp_dex_src += true_B_gap;
        }
        dex_src += 16;
    }
}



void _THREAD_FUNCTION_ decx::_avx256_sort_ST_MatB_w8(float* srcB, float* dstB, const int4& dims_pkg)
{
    size_t dex_src = 0, dex_dst = 0, tmp_dex_src = 0,
        true_B_gap = (size_t)dims_pkg.z - 8;

    for (int i = 0; i < dims_pkg.y / 16; ++i)
    {
        tmp_dex_src = dex_src;
        for (int _L = 0; _L < dims_pkg.x; ++_L)
        {
            _mm256_storeu_ps(dstB + dex_dst, _mm256_loadu_ps(srcB + tmp_dex_src));
            dex_dst += 8;
            tmp_dex_src += 8;
            _mm256_storeu_ps(dstB + dex_dst, _mm256_loadu_ps(srcB + tmp_dex_src));
            dex_dst += 8;
            tmp_dex_src += true_B_gap;
        }
        dex_src += 16;
    }
    tmp_dex_src = dex_src;
    for (int _L = 0; _L < dims_pkg.x; ++_L)
    {
        _mm256_storeu_ps(dstB + dex_dst, _mm256_loadu_ps(srcB + tmp_dex_src));
        dex_dst += 8;
        tmp_dex_src += (size_t)dims_pkg.z;
    }
}



template <int thread_num>
void decx::sort_MatB_16(float* srcB, float* dstB, const int2& dims_pkg)
{
    std::future<void>* __async_stream = new std::future<void>[16];

    int lane_num = dims_pkg.y / 16;

    int4 proc_dims = make_int4(dims_pkg.x, (lane_num / thread_num) * 16, dims_pkg.y, 0);
    size_t B_frag = (size_t)dims_pkg.x * (size_t)proc_dims.y;
    int4 proc_dim_last = make_int4(dims_pkg.x, ((lane_num / thread_num) + lane_num % thread_num) * 16, dims_pkg.y, 0);

    for (int i = 0; i < thread_num - 1; ++i) {
        __async_stream[i] = decx::thread_pool.register_task(
            _avx256_sort_ST_MatB,
            srcB + i * (size_t)proc_dims.y,
            dstB + i * B_frag,
            proc_dims);
    }
    __async_stream[thread_num - 1] = decx::thread_pool.register_task(
        _avx256_sort_ST_MatB,
        srcB + (thread_num - 1) * (size_t)proc_dims.y,
        dstB + (thread_num - 1) * B_frag,
        proc_dim_last);

    for (int i = 0; i < thread_num; ++i) {
        __async_stream[i].get();
    }
}




template <int thread_num>
void decx::sort_MatB_16_w8(float* srcB, float* dstB, const int2& dims_pkg)
{
    std::future<void>* __async_stream = new std::future<void>[thread_num];

    int lane_num = (dims_pkg.y - 8) / 16;

    int4 proc_dims = make_int4(dims_pkg.x, (lane_num / thread_num) * 16, dims_pkg.y, 0);
    size_t B_frag = (size_t)dims_pkg.x * (size_t)proc_dims.y;
    int4 proc_dim_last = make_int4(dims_pkg.x, ((lane_num / thread_num) + lane_num % thread_num) * 16 + 8, dims_pkg.y, 0);

    for (int i = 0; i < thread_num - 1; ++i) {
        __async_stream[i] = decx::thread_pool.register_task(
            _avx256_sort_ST_MatB,
            srcB + i * (size_t)proc_dims.y,
            dstB + i * B_frag,
            proc_dims);
    }
    __async_stream[thread_num - 1] = decx::thread_pool.register_task(
        _avx256_sort_ST_MatB_w8,
        srcB + (thread_num - 1) * (size_t)proc_dims.y,
        dstB + (thread_num - 1) * B_frag,
        proc_dim_last);

    for (int i = 0; i < thread_num; ++i) {
        __async_stream[i].get();
    }
}

// ----------------------------- end sort ---------------------------------------


#endif