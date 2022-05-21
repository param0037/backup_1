/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
* 
*    cvt_colors uses CPU and avx instead of CUDA, since it does not need much
* calculation. This module call the APIs from DECX_cpu.dll(.so)
*/

#pragma once

#include "../../core/thread_management/thread_pool.h"
#include "../cv_classes/cv_classes.h"
#include "cvt_colors_def.h"


#ifndef _CVT_COLORS_H_
#define _CVT_COLORS_H_



#define _BGR2GRAY_UNIT_(__ord, __dex) {                                \
reg_pixel.m128_f32[0] = (float)reg_tmp.m128i_u8[__dex * 4 + 0];        \
reg_pixel.m128_f32[1] = (float)reg_tmp.m128i_u8[__dex * 4 + 1];        \
reg_pixel.m128_f32[2] = (float)reg_tmp.m128i_u8[__dex * 4 + 2];        \
                                                                    \
reg_pixel = _mm_mul_ps(reg_pixel, scalar);                            \
reg_pixel.m128_f32[0] += reg_pixel.m128_f32[1];                        \
reg_pixel.m128_f32[0] += reg_pixel.m128_f32[2];                        \
                                                                    \
reg_ans.m128i_u8[__ord * 4 + __dex] = (uchar)reg_pixel.m128_f32[0];    \
}



#define _PRESERVE_B_UNIT_(__ord, __dex) {                                \
reg_ans.m128i_u8[__ord * 4 + __dex] = reg_tmp.m128i_u8[__dex * 4 + 0];    \
}


#define _PRESERVE_G_UNIT_(__ord, __dex) {                                \
reg_ans.m128i_u8[__ord * 4 + __dex] = reg_tmp.m128i_u8[__dex * 4 + 1];    \
}


#define _PRESERVE_R_UNIT_(__ord, __dex) {                                \
reg_ans.m128i_u8[__ord * 4 + __dex] = reg_tmp.m128i_u8[__dex * 4 + 2];    \
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
}



static void _THREAD_FUNCTION_ decx::_BGR2Gray_ST_UC2UC(float* src, float* dst, const int2 dims)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    __m128i reg_tmp, reg_ans;
    __m128 reg_pixel,
        scalar = _mm_set_ps(0.114f, 0.587f, 0.299f, 0);

    for (int i = 0; i < dims.y; ++i) {
        for (int j = 0; j < dims.x; ++j) {
            *((__m128*)&reg_tmp) = _mm_load_ps(src + glo_dex_src);
            _BGR2GRAY_UNIT_(0, 0);
            _BGR2GRAY_UNIT_(0, 1);
            _BGR2GRAY_UNIT_(0, 2);
            _BGR2GRAY_UNIT_(0, 3);
            glo_dex_src += 4;

            *((__m128*) & reg_tmp) = _mm_load_ps(src + glo_dex_src);
            _BGR2GRAY_UNIT_(1, 0);
            _BGR2GRAY_UNIT_(1, 1);
            _BGR2GRAY_UNIT_(1, 2);
            _BGR2GRAY_UNIT_(1, 3);
            glo_dex_src += 4;

            *((__m128*) & reg_tmp) = _mm_load_ps(src + glo_dex_src);
            _BGR2GRAY_UNIT_(2, 0);
            _BGR2GRAY_UNIT_(2, 1);
            _BGR2GRAY_UNIT_(2, 2);
            _BGR2GRAY_UNIT_(2, 3);
            glo_dex_src += 4;

            *((__m128*) & reg_tmp) = _mm_load_ps(src + glo_dex_src);
            _BGR2GRAY_UNIT_(3, 0);
            _BGR2GRAY_UNIT_(3, 1);
            _BGR2GRAY_UNIT_(3, 2);
            _BGR2GRAY_UNIT_(3, 3);
            glo_dex_src += 4;

            _mm_store_ps(dst + glo_dex_dst, *((__m128*)&reg_ans));
            glo_dex_dst += 4;
        }
    }
}



static void _THREAD_FUNCTION_ decx::_Preserve_B_ST_UC2UC(float* src, float* dst, const int2 dims)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    __m128i reg_tmp, reg_ans;

    for (int i = 0; i < dims.y; ++i) {
        for (int j = 0; j < dims.x; ++j) {
            *((__m128*) & reg_tmp) = _mm_load_ps(src + glo_dex_src);
            _PRESERVE_B_UNIT_(0, 0);
            _PRESERVE_B_UNIT_(0, 1);
            _PRESERVE_B_UNIT_(0, 2);
            _PRESERVE_B_UNIT_(0, 3);
            glo_dex_src += 4;

            *((__m128*) & reg_tmp) = _mm_load_ps(src + glo_dex_src);
            _PRESERVE_B_UNIT_(1, 0);
            _PRESERVE_B_UNIT_(1, 1);
            _PRESERVE_B_UNIT_(1, 2);
            _PRESERVE_B_UNIT_(1, 3);
            glo_dex_src += 4;

            *((__m128*) & reg_tmp) = _mm_load_ps(src + glo_dex_src);
            _PRESERVE_B_UNIT_(2, 0);
            _PRESERVE_B_UNIT_(2, 1);
            _PRESERVE_B_UNIT_(2, 2);
            _PRESERVE_B_UNIT_(2, 3);
            glo_dex_src += 4;

            *((__m128*) & reg_tmp) = _mm_load_ps(src + glo_dex_src);
            _PRESERVE_B_UNIT_(3, 0);
            _PRESERVE_B_UNIT_(3, 1);
            _PRESERVE_B_UNIT_(3, 2);
            _PRESERVE_B_UNIT_(3, 3);
            glo_dex_src += 4;

            _mm_store_ps(dst + glo_dex_dst, *((__m128*) & reg_ans));
            glo_dex_dst += 4;
        }
    }
}



static void _THREAD_FUNCTION_ decx::_Preserve_G_ST_UC2UC(float* src, float* dst, const int2 dims)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    __m128i reg_tmp, reg_ans;

    for (int i = 0; i < dims.y; ++i) {
        for (int j = 0; j < dims.x; ++j) {
            *((__m128*) & reg_tmp) = _mm_load_ps(src + glo_dex_src);
            _PRESERVE_G_UNIT_(0, 0);
            _PRESERVE_G_UNIT_(0, 1);
            _PRESERVE_G_UNIT_(0, 2);
            _PRESERVE_G_UNIT_(0, 3);
            glo_dex_src += 4;

            *((__m128*) & reg_tmp) = _mm_load_ps(src + glo_dex_src);
            _PRESERVE_G_UNIT_(1, 0);
            _PRESERVE_G_UNIT_(1, 1);
            _PRESERVE_G_UNIT_(1, 2);
            _PRESERVE_G_UNIT_(1, 3);
            glo_dex_src += 4;

            *((__m128*) & reg_tmp) = _mm_load_ps(src + glo_dex_src);
            _PRESERVE_G_UNIT_(2, 0);
            _PRESERVE_G_UNIT_(2, 1);
            _PRESERVE_G_UNIT_(2, 2);
            _PRESERVE_G_UNIT_(2, 3);
            glo_dex_src += 4;

            *((__m128*) & reg_tmp) = _mm_load_ps(src + glo_dex_src);
            _PRESERVE_G_UNIT_(3, 0);
            _PRESERVE_G_UNIT_(3, 1);
            _PRESERVE_G_UNIT_(3, 2);
            _PRESERVE_G_UNIT_(3, 3);
            glo_dex_src += 4;

            _mm_store_ps(dst + glo_dex_dst, *((__m128*) & reg_ans));
            glo_dex_dst += 4;
        }
    }
}



static void _THREAD_FUNCTION_ decx::_Preserve_R_ST_UC2UC(float* src, float* dst, const int2 dims)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    __m128i reg_tmp, reg_ans;

    for (int i = 0; i < dims.y; ++i) {
        for (int j = 0; j < dims.x; ++j) {
            *((__m128*) & reg_tmp) = _mm_load_ps(src + glo_dex_src);
            _PRESERVE_R_UNIT_(0, 0);
            _PRESERVE_R_UNIT_(0, 1);
            _PRESERVE_R_UNIT_(0, 2);
            _PRESERVE_R_UNIT_(0, 3);
            glo_dex_src += 4;

            *((__m128*) & reg_tmp) = _mm_load_ps(src + glo_dex_src);
            _PRESERVE_R_UNIT_(1, 0);
            _PRESERVE_R_UNIT_(1, 1);
            _PRESERVE_R_UNIT_(1, 2);
            _PRESERVE_R_UNIT_(1, 3);
            glo_dex_src += 4;

            *((__m128*) & reg_tmp) = _mm_load_ps(src + glo_dex_src);
            _PRESERVE_R_UNIT_(2, 0);
            _PRESERVE_R_UNIT_(2, 1);
            _PRESERVE_R_UNIT_(2, 2);
            _PRESERVE_R_UNIT_(2, 3);
            glo_dex_src += 4;

            *((__m128*) & reg_tmp) = _mm_load_ps(src + glo_dex_src);
            _PRESERVE_R_UNIT_(3, 0);
            _PRESERVE_R_UNIT_(3, 1);
            _PRESERVE_R_UNIT_(3, 2);
            _PRESERVE_R_UNIT_(3, 3);
            glo_dex_src += 4;

            _mm_store_ps(dst + glo_dex_dst, *((__m128*) & reg_ans));
            glo_dex_dst += 4;
        }
    }
}


// --------------------------------------- CALLERS --------------------------------------------------------


void decx::_BGR2Gray_UC2UC_caller(float* src, float* dst, const int2 dims)
{
    int _concurrent = (int)decx::thread_pool._hardware_concurrent;
    int2 sub_dims = make_int2(dims.x / 16, dims.y / _concurrent);
    size_t fragment = (size_t)sub_dims.x * (size_t)sub_dims.y, offset = 0;
    decx::PtrInfo<std::future<void>> _thread_handle;

    decx::alloc::_host_virtual_page_malloc<std::future<void>>(&_thread_handle, _concurrent * sizeof(std::future<void>));

    for (int i = 0; i < _concurrent - 1; ++i) {
        _thread_handle.ptr[i] = decx::thread_pool.register_task(decx::_BGR2Gray_ST_UC2UC, src + (offset << 4), dst + (offset << 2), sub_dims);
        offset += fragment;
    }

    sub_dims.y = dims.y - (_concurrent - 1) * sub_dims.y;
    _thread_handle.ptr[decx::thread_pool._hardware_concurrent - 1] = 
        decx::thread_pool.register_task(decx::_BGR2Gray_ST_UC2UC, src + (offset << 4), dst + (offset << 2), sub_dims);
    
    for (int i = 0; i < _concurrent; ++i) {
        _thread_handle.ptr[i].get();
    }

    decx::alloc::_host_virtual_page_dealloc<std::future<void>>(&_thread_handle);
}



void decx::_Preserve_B_UC2UC_caller(float* src, float* dst, const int2 dims)
{
    int _concurrent = (int)decx::thread_pool._hardware_concurrent;
    int2 sub_dims = make_int2(dims.x / 16, dims.y / _concurrent);
    size_t fragment = (size_t)sub_dims.x * (size_t)sub_dims.y, offset = 0;
    decx::PtrInfo<std::future<void>> _thread_handle;

    decx::alloc::_host_virtual_page_malloc<std::future<void>>(&_thread_handle, _concurrent * sizeof(std::future<void>));

    for (int i = 0; i < _concurrent - 1; ++i) {
        _thread_handle.ptr[i] = decx::thread_pool.register_task(decx::_Preserve_B_ST_UC2UC, src + (offset << 4), dst + (offset << 2), sub_dims);
        offset += fragment;
    }

    sub_dims.y = dims.y - (_concurrent - 1) * sub_dims.y;
    _thread_handle.ptr[decx::thread_pool._hardware_concurrent - 1] =
        decx::thread_pool.register_task(decx::_Preserve_B_ST_UC2UC, src + (offset << 4), dst + (offset << 2), sub_dims);

    for (int i = 0; i < _concurrent; ++i) {
        _thread_handle.ptr[i].get();
    }

    decx::alloc::_host_virtual_page_dealloc<std::future<void>>(&_thread_handle);
}



void decx::_Preserve_G_UC2UC_caller(float* src, float* dst, const int2 dims)
{
    int _concurrent = (int)decx::thread_pool._hardware_concurrent;
    int2 sub_dims = make_int2(dims.x / 16, dims.y / _concurrent);
    size_t fragment = (size_t)sub_dims.x * (size_t)sub_dims.y, offset = 0;
    decx::PtrInfo<std::future<void>> _thread_handle;

    decx::alloc::_host_virtual_page_malloc<std::future<void>>(&_thread_handle, _concurrent * sizeof(std::future<void>));

    for (int i = 0; i < _concurrent - 1; ++i) {
        _thread_handle.ptr[i] = decx::thread_pool.register_task(decx::_Preserve_G_ST_UC2UC, src + (offset << 4), dst + (offset << 2), sub_dims);
        offset += fragment;
    }

    sub_dims.y = dims.y - (_concurrent - 1) * sub_dims.y;
    _thread_handle.ptr[decx::thread_pool._hardware_concurrent - 1] =
        decx::thread_pool.register_task(decx::_Preserve_G_ST_UC2UC, src + (offset << 4), dst + (offset << 2), sub_dims);

    for (int i = 0; i < _concurrent; ++i) {
        _thread_handle.ptr[i].get();
    }

    decx::alloc::_host_virtual_page_dealloc<std::future<void>>(&_thread_handle);
}



void decx::_Preserve_R_UC2UC_caller(float* src, float* dst, const int2 dims)
{
    int _concurrent = (int)decx::thread_pool._hardware_concurrent;
    int2 sub_dims = make_int2(dims.x / 16, dims.y / _concurrent);
    size_t fragment = (size_t)sub_dims.x * (size_t)sub_dims.y, offset = 0;
    decx::PtrInfo<std::future<void>> _thread_handle;

    decx::alloc::_host_virtual_page_malloc<std::future<void>>(&_thread_handle, _concurrent * sizeof(std::future<void>));

    for (int i = 0; i < _concurrent - 1; ++i) {
        _thread_handle.ptr[i] = decx::thread_pool.register_task(decx::_Preserve_R_ST_UC2UC, src + (offset << 4), dst + (offset << 2), sub_dims);
        offset += fragment;
    }

    sub_dims.y = dims.y - (_concurrent - 1) * sub_dims.y;
    _thread_handle.ptr[decx::thread_pool._hardware_concurrent - 1] =
        decx::thread_pool.register_task(decx::_Preserve_R_ST_UC2UC, src + (offset << 4), dst + (offset << 2), sub_dims);

    for (int i = 0; i < _concurrent; ++i) {
        _thread_handle.ptr[i].get();
    }

    decx::alloc::_host_virtual_page_dealloc<std::future<void>>(&_thread_handle);
}


#endif