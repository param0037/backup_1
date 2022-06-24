/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
* 
*    cvt_colors uses CPU and avx instead of CUDA, since it does not need much
* calculation. This module call the APIs from DECX_cpu.dll(.so)
*/

#pragma once

#ifdef _DECX_CPU_CODES_
#include "../../core/thread_management/thread_pool.h"
#include "../cv_classes/cv_classes.h"


#ifndef _CVT_COLORS_H_
#define _CVT_COLORS_H_



namespace decx
{
    /*
    * dims.x : It is the true pitch of pixel data matrix, scale of uchar
    * dims.y : It is the height of pixel data matrix
    */
    _DECX_API_ void _BGR2Gray_UC2UC_caller(float* src, float* dst, const int2 dims);

    /*
    * dims.x : It is the true pitch of pixel data matrix, scale of uchar
    * dims.y : It is the height of pixel data matrix
    */
    _DECX_API_ void _Preserve_B_UC2UC_caller(float* src, float* dst, const int2 dims);

    /*
    * dims.x : It is the true pitch of pixel data matrix, scale of uchar
    * dims.y : It is the height of pixel data matrix
    */
    _DECX_API_ void _Preserve_G_UC2UC_caller(float* src, float* dst, const int2 dims);

    /*
    * dims.x : It is the true pitch of pixel data matrix, scale of uchar
    * dims.y : It is the height of pixel data matrix
    */
    _DECX_API_ void _Preserve_R_UC2UC_caller(float* src, float* dst, const int2 dims);


    /*
    * dims.x : It is the true pitch of pixel data matrix, scale of uchar
    * dims.y : It is the height of pixel data matrix
    */
    _DECX_API_ void _Preserve_A_UC2UC_caller(float* src, float* dst, const int2 dims);
}



namespace decx
{
    /*
    * src and dst have two different scale of pitch, take dst's as scale, which is 16x
    * dims : The dims info of processed area, dims.x : pitch(16x); dims.y : height
    */
    static void _THREAD_FUNCTION_ _BGR2Gray_ST_UC2UC(float* src, float* dst, const int2 dims);


    /*
    * src and dst have two different scale of pitch, take dst's as scale, which is 16x
    * dims : The dims info of processed area, dims.x : pitch(16x); dims.y : height
    */
    static void _THREAD_FUNCTION_ _Preserve_B_ST_UC2UC(float* src, float* dst, const int2 dims);

    /*
    * src and dst have two different scale of pitch, take dst's as scale, which is 16x
    * dims : The dims info of processed area, dims.x : pitch(16x); dims.y : height
    */
    static void _THREAD_FUNCTION_ _Preserve_G_ST_UC2UC(float* src, float* dst, const int2 dims);

    /*
    * src and dst have two different scale of pitch, take dst's as scale, which is 16x
    * dims : The dims info of processed area, dims.x : pitch(16x); dims.y : height
    */
    static void _THREAD_FUNCTION_ _Preserve_R_ST_UC2UC(float* src, float* dst, const int2 dims);

    /*
    * src and dst have two different scale of pitch, take dst's as scale, which is 16x
    * dims : The dims info of processed area, dims.x : pitch(16x); dims.y : height
    */
    static void _THREAD_FUNCTION_ _Preserve_A_ST_UC2UC(float* src, float* dst, const int2 dims);
}



static void _THREAD_FUNCTION_ decx::_BGR2Gray_ST_UC2UC(float* src, float* dst, const int2 dims)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    __m128i __recv, __recv_cvt0;
    __m128 pixel_f, __recv_cvt1,
        scalar = _mm_set_ps(0, 0.299f, 0.587f, 0.114f);

    float res;

    uchar4 reg_dst;

    for (int i = 0; i < dims.y; ++i) {
        for (int j = 0; j < dims.x; ++j) {
            *((__m128*)&__recv) = _mm_load_ps(src + glo_dex_src);
            glo_dex_src += 4;

            ((float*)&__recv_cvt0)[0] = ((float*)&__recv)[0];
            *((__m128i*)&__recv_cvt1) = _mm_cvtepi8_epi32(__recv_cvt0);
            pixel_f = _mm_cvtepi32_ps(*((__m128i*)&__recv_cvt1));

            __recv_cvt1 = _mm_mul_ps(pixel_f, scalar);

            res = decx::utils::_mm128_h_sum(__recv_cvt1);
            reg_dst.x = (uchar)res;

            ((float*)&__recv_cvt0)[0] = ((float*)&__recv)[1];
            *((__m128i*) & __recv_cvt1) = _mm_cvtepi8_epi32(__recv_cvt0);
            pixel_f = _mm_cvtepi32_ps(*((__m128i*) & __recv_cvt1));

            __recv_cvt1 = _mm_mul_ps(pixel_f, scalar);

            res = decx::utils::_mm128_h_sum(__recv_cvt1);
            reg_dst.y = (uchar)res;

            ((float*)&__recv_cvt0)[0] = ((float*)&__recv)[2];
            *((__m128i*) & __recv_cvt1) = _mm_cvtepi8_epi32(__recv_cvt0);
            pixel_f = _mm_cvtepi32_ps(*((__m128i*) & __recv_cvt1));

            __recv_cvt1 = _mm_mul_ps(pixel_f, scalar);

            res = decx::utils::_mm128_h_sum(__recv_cvt1);
            reg_dst.z = (uchar)res;

            ((float*)&__recv_cvt0)[0] = ((float*)&__recv)[3];
            *((__m128i*) & __recv_cvt1) = _mm_cvtepi8_epi32(__recv_cvt0);
            pixel_f = _mm_cvtepi32_ps(*((__m128i*) & __recv_cvt1));

            __recv_cvt1 = _mm_mul_ps(pixel_f, scalar);

            res = decx::utils::_mm128_h_sum(__recv_cvt1);
            reg_dst.w = (uchar)res;

            dst[glo_dex_dst] = *((float*)&reg_dst);

            ++glo_dex_dst;
        }
    }
}



static void _THREAD_FUNCTION_ decx::_Preserve_B_ST_UC2UC(float* src, float* dst, const int2 dims)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    __m128i __recv;
    
    uchar4 reg_dst;

    for (int i = 0; i < dims.y; ++i) {
        for (int j = 0; j < dims.x; ++j) {
            *((__m128*) & __recv) = _mm_load_ps(src + glo_dex_src);
            glo_dex_src += 4;

            reg_dst.x = ((uchar4*)&__recv)[0].x;
            reg_dst.y = ((uchar4*)&__recv)[1].x;
            reg_dst.z = ((uchar4*)&__recv)[2].x;
            reg_dst.w = ((uchar4*)&__recv)[3].x;

            dst[glo_dex_dst] = *((float*)&reg_dst);

            ++glo_dex_dst;
        }
    }
}



static void _THREAD_FUNCTION_ decx::_Preserve_G_ST_UC2UC(float* src, float* dst, const int2 dims)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    __m128i __recv;

    uchar4 reg_dst;

    for (int i = 0; i < dims.y; ++i) {
        for (int j = 0; j < dims.x; ++j) {
            *((__m128*) & __recv) = _mm_load_ps(src + glo_dex_src);
            glo_dex_src += 4;

            reg_dst.x = ((uchar4*)&__recv)[0].y;
            reg_dst.y = ((uchar4*)&__recv)[1].y;
            reg_dst.z = ((uchar4*)&__recv)[2].y;
            reg_dst.w = ((uchar4*)&__recv)[3].y;

            dst[glo_dex_dst] = *((float*)&reg_dst);

            ++glo_dex_dst;
        }
    }
}



static void _THREAD_FUNCTION_ decx::_Preserve_R_ST_UC2UC(float* src, float* dst, const int2 dims)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    __m128i __recv;

    uchar4 reg_dst;

    for (int i = 0; i < dims.y; ++i) {
        for (int j = 0; j < dims.x; ++j) {
            *((__m128*) & __recv) = _mm_load_ps(src + glo_dex_src);
            glo_dex_src += 4;

            reg_dst.x = ((uchar4*)&__recv)[0].z;
            reg_dst.y = ((uchar4*)&__recv)[1].z;
            reg_dst.z = ((uchar4*)&__recv)[2].z;
            reg_dst.w = ((uchar4*)&__recv)[3].z;

            dst[glo_dex_dst] = *((float*)&reg_dst);

            ++glo_dex_dst;
        }
    }
}



static void _THREAD_FUNCTION_ decx::_Preserve_A_ST_UC2UC(float* src, float* dst, const int2 dims)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    __m128i __recv;

    uchar4 reg_dst;

    for (int i = 0; i < dims.y; ++i) {
        for (int j = 0; j < dims.x; ++j) {
            *((__m128*) & __recv) = _mm_load_ps(src + glo_dex_src);
            glo_dex_src += 4;

            reg_dst.x = ((uchar4*)&__recv)[0].w;
            reg_dst.y = ((uchar4*)&__recv)[1].w;
            reg_dst.z = ((uchar4*)&__recv)[2].w;
            reg_dst.w = ((uchar4*)&__recv)[3].w;

            dst[glo_dex_dst] = *((float*)&reg_dst);

            ++glo_dex_dst;
        }
    }
}


// --------------------------------------- CALLERS --------------------------------------------------------


void decx::_BGR2Gray_UC2UC_caller(float* src, float* dst, const int2 dims)
{
    int _concurrent = (int)decx::thread_pool._hardware_concurrent;
    int2 sub_dims = make_int2(dims.x / 4, dims.y / _concurrent);
    size_t fragment = (size_t)sub_dims.x * (size_t)sub_dims.y, offset = 0;

    std::future<void>* _thread_handle = new std::future<void>[12];

    for (int i = 0; i < _concurrent - 1; ++i) {
        _thread_handle[i] = decx::thread_pool.register_task(decx::_BGR2Gray_ST_UC2UC, src + (offset << 2), dst + offset, sub_dims);
        offset += fragment;
    }

    sub_dims.y = dims.y - (_concurrent - 1) * sub_dims.y;
    _thread_handle[decx::thread_pool._hardware_concurrent - 1] = 
        decx::thread_pool.register_task(decx::_BGR2Gray_ST_UC2UC, src + (offset << 2), dst + offset, sub_dims);
    
    for (int i = 0; i < _concurrent; ++i) {
        _thread_handle[i].get();
    }

    delete[] _thread_handle;
}





void decx::_Preserve_B_UC2UC_caller(float* src, float* dst, const int2 dims)
{
    int _concurrent = (int)decx::thread_pool._hardware_concurrent;
    int2 sub_dims = make_int2(dims.x / 4, dims.y / _concurrent);
    size_t fragment = (size_t)sub_dims.x * (size_t)sub_dims.y, offset = 0;

    std::future<void>* _thread_handle = new std::future<void>[12];

    for (int i = 0; i < _concurrent - 1; ++i) {
        _thread_handle[i] = decx::thread_pool.register_task(decx::_Preserve_B_ST_UC2UC, src + (offset << 2), dst + offset, sub_dims);
        offset += fragment;
    }

    sub_dims.y = dims.y - (_concurrent - 1) * sub_dims.y;
    _thread_handle[decx::thread_pool._hardware_concurrent - 1] =
        decx::thread_pool.register_task(decx::_Preserve_B_ST_UC2UC, src + (offset << 2), dst + offset, sub_dims);

    for (int i = 0; i < _concurrent; ++i) {
        _thread_handle[i].get();
    }

    delete[] _thread_handle;
}



void decx::_Preserve_G_UC2UC_caller(float* src, float* dst, const int2 dims)
{
    int _concurrent = (int)decx::thread_pool._hardware_concurrent;
    int2 sub_dims = make_int2(dims.x / 4, dims.y / _concurrent);
    size_t fragment = (size_t)sub_dims.x * (size_t)sub_dims.y, offset = 0;

    std::future<void>* _thread_handle = new std::future<void>[12];

    for (int i = 0; i < _concurrent - 1; ++i) {
        _thread_handle[i] = decx::thread_pool.register_task(decx::_Preserve_G_ST_UC2UC, src + (offset << 2), dst + offset, sub_dims);
        offset += fragment;
    }

    sub_dims.y = dims.y - (_concurrent - 1) * sub_dims.y;
    _thread_handle[decx::thread_pool._hardware_concurrent - 1] =
        decx::thread_pool.register_task(decx::_Preserve_G_ST_UC2UC, src + (offset << 2), dst + offset, sub_dims);

    for (int i = 0; i < _concurrent; ++i) {
        _thread_handle[i].get();
    }

    delete[] _thread_handle;
}



void decx::_Preserve_R_UC2UC_caller(float* src, float* dst, const int2 dims)
{
    int _concurrent = (int)decx::thread_pool._hardware_concurrent;
    int2 sub_dims = make_int2(dims.x / 4, dims.y / _concurrent);
    size_t fragment = (size_t)sub_dims.x * (size_t)sub_dims.y, offset = 0;

    std::future<void>* _thread_handle = new std::future<void>[12];

    for (int i = 0; i < _concurrent - 1; ++i) {
        _thread_handle[i] = decx::thread_pool.register_task(decx::_Preserve_R_ST_UC2UC, src + (offset << 2), dst + offset, sub_dims);
        offset += fragment;
    }

    sub_dims.y = dims.y - (_concurrent - 1) * sub_dims.y;
    _thread_handle[decx::thread_pool._hardware_concurrent - 1] =
        decx::thread_pool.register_task(decx::_Preserve_R_ST_UC2UC, src + (offset << 2), dst + offset, sub_dims);

    for (int i = 0; i < _concurrent; ++i) {
        _thread_handle[i].get();
    }

    delete[] _thread_handle;
}


void decx::_Preserve_A_UC2UC_caller(float* src, float* dst, const int2 dims)
{
    int _concurrent = (int)decx::thread_pool._hardware_concurrent;
    int2 sub_dims = make_int2(dims.x / 4, dims.y / _concurrent);
    size_t fragment = (size_t)sub_dims.x * (size_t)sub_dims.y, offset = 0;

    std::future<void>* _thread_handle = new std::future<void>[12];

    for (int i = 0; i < _concurrent - 1; ++i) {
        _thread_handle[i] = decx::thread_pool.register_task(decx::_Preserve_A_ST_UC2UC, src + (offset << 2), dst + offset, sub_dims);
        offset += fragment;
    }

    sub_dims.y = dims.y - (_concurrent - 1) * sub_dims.y;
    _thread_handle[decx::thread_pool._hardware_concurrent - 1] =
        decx::thread_pool.register_task(decx::_Preserve_A_ST_UC2UC, src + (offset << 2), dst + offset, sub_dims);

    for (int i = 0; i < _concurrent; ++i) {
        _thread_handle[i].get();
    }

    delete[] _thread_handle;
}


#endif

#endif
