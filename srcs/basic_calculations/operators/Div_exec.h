/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _DIV_EXEC_H_
#define _DIV_EXEC_H_

#include "../../core/basic.h"
#include "../../core/thread_management/thread_pool.h"
#include "../../classes/classes_util.h"


namespace decx
{
    /**
    * @param A : pointer of sub-matrix A
    * @param B : pointer of sub-matrix B
    * @param dst : pointer of sub-matrix dst
    * @param len : regard the data space as a 1D array, the length is in float8
    */
    void _THREAD_FUNCTION_ div_m_fvec8_ST(float* A, float* B, float* dst, size_t len);


    void _THREAD_FUNCTION_ div_m_ivec8_ST(__m256i* A, __m256i* B, __m256i* dst, size_t len);


    void _THREAD_FUNCTION_ div_m_dvec4_ST(double* A, double* B, double* dst, size_t len);


    void _THREAD_FUNCTION_ div_c_fvec8_ST(float* src, const float __x, float* dst, size_t len);


    void _THREAD_FUNCTION_ div_c_ivec8_ST(__m256i* src, const int __x, __m256i* dst, size_t len);


    void _THREAD_FUNCTION_ div_c_dvec4_ST(double* src, const double __x, double* dst, size_t len);


    void _THREAD_FUNCTION_ div_cinv_fvec8_ST(float* src, const float __x, float* dst, size_t len);


    void _THREAD_FUNCTION_ div_cinv_ivec8_ST(__m256i* src, const int __x, __m256i* dst, size_t len);


    void _THREAD_FUNCTION_ div_cinv_dvec4_ST(double* src, const double __x, double* dst, size_t len);



    /**
    * @param A : pointer of sub-matrix A
    * @param B : pointer of sub-matrix B
    * @param dst : pointer of sub-matrix dst
    * @param len : regard the data space as a 1D array, the length is in float
    */
    void Kdiv_m(float* A, float* B, float* dst, const size_t len);


    void Kdiv_m(int* A, int* B, int* dst, const size_t len);


    void Kdiv_m(double* A, double* B, double* dst, const size_t len);


    void Kdiv_c(float* src, const float __x, float* dst, const size_t len);


    void Kdiv_c(int* src, const int __x, int* dst, const size_t len);


    void Kdiv_c(double* src, const double __x, double* dst, const size_t len);


    void Kdiv_cinv(float* src, const float __x, float* dst, const size_t len);


    void Kdiv_cinv(int* src, const int __x, int* dst, const size_t len);


    void Kdiv_cinv(double* src, const double __x, double* dst, const size_t len);
}


void _THREAD_FUNCTION_ decx::div_m_fvec8_ST(float* A, float* B, float* dst, size_t len)
{
    __m256 tmpA, tmpB, tmpdst;
    for (uint i = 0; i < len; ++i){
        tmpA = _mm256_load_ps(A + (i << 3));
        tmpB = _mm256_load_ps(B + (i << 3));

        tmpdst = _mm256_div_ps(tmpA, tmpB);

        _mm256_store_ps(dst + (i << 3), tmpdst);
    }
}


void _THREAD_FUNCTION_ decx::div_m_ivec8_ST(__m256i* A, __m256i* B, __m256i* dst, size_t len)
{
    __m256 tmpA, tmpB, tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_cvtepi32_ps(_mm256_load_si256(A + i));
        tmpB = _mm256_cvtepi32_ps(_mm256_load_si256(B + i));

        tmpdst = _mm256_div_ps(tmpA, tmpB);

        _mm256_store_si256(dst + i, _mm256_cvtps_epi32(tmpdst));
    }
}


void _THREAD_FUNCTION_ decx::div_m_dvec4_ST(double* A, double* B, double* dst, size_t len)
{
    __m256d tmpA, tmpB, tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_load_pd(A + (i << 2));
        tmpB = _mm256_load_pd(B + (i << 2));

        tmpdst = _mm256_div_pd(tmpA, tmpB);

        _mm256_store_pd(dst + (i << 2), tmpdst);
    }
}



void _THREAD_FUNCTION_ decx::div_c_fvec8_ST(float* src, const float __x, float* dst, size_t len)
{
    __m256 tmpsrc, tmpX = _mm256_set1_ps(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_ps(src + (i << 3));

        tmpdst = _mm256_div_ps(tmpsrc, tmpX);

        _mm256_store_ps(dst + (i << 3), tmpdst);
    }
}


void _THREAD_FUNCTION_ decx::div_c_ivec8_ST(__m256i* src, const int __x, __m256i* dst, size_t len)
{
    __m256 tmpsrc, tmpX = _mm256_set1_ps(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_cvtepi32_ps(_mm256_load_si256(src + i));

        tmpdst = _mm256_div_ps(tmpsrc, tmpX);

        _mm256_store_si256(dst + i, _mm256_cvtps_epi32(tmpdst));
    }
}


void _THREAD_FUNCTION_ decx::div_c_dvec4_ST(double* src, const double __x, double* dst, size_t len)
{
    __m256d tmpsrc, tmpX = _mm256_set1_pd(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_pd(src + (i << 2));

        tmpdst = _mm256_div_pd(tmpsrc, tmpX);

        _mm256_store_pd(dst + (i << 2), tmpdst);
    }
}


void _THREAD_FUNCTION_ decx::div_cinv_fvec8_ST(float* src, const float __x, float* dst, size_t len)
{
    __m256 tmpsrc, tmpX = _mm256_set1_ps(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_ps(src + (i << 3));

        tmpdst = _mm256_div_ps(tmpX, tmpsrc);

        _mm256_store_ps(dst + (i << 3), tmpdst);
    }
}


void _THREAD_FUNCTION_ decx::div_cinv_ivec8_ST(__m256i* src, const int __x, __m256i* dst, size_t len)
{
    __m256i tmpsrc, tmpX = _mm256_set1_epi32(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_si256(src + i);

        tmpdst = _mm256_div_epi32(tmpX, tmpsrc);

        _mm256_store_si256(dst + i, tmpdst);
    }
}


void _THREAD_FUNCTION_ decx::div_cinv_dvec4_ST(double* src, const double __x, double* dst, size_t len)
{
    __m256d tmpsrc, tmpX = _mm256_set1_pd(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_pd(src + (i << 2));

        tmpdst = _mm256_div_pd(tmpX, tmpsrc);

        _mm256_store_pd(dst + (i << 2), tmpdst);
    }
}


// ----------------------------------------- callers -----------------------------------------------------------


void decx::Kdiv_m(float* A, float* B, float* dst, const size_t len)
{
    const uint thread_num = decx::cpI.cpu_concurrency;

    decx::utils::_thr_1D t_arrange_info(thread_num, len / 8);

    std::future<void>* __async_stream = new std::future<void>[thread_num];

    if (t_arrange_info.is_avg) {
        for (int i = 0; i < thread_num; ++i) {
            __async_stream[i] = 
                decx::thread_pool.register_task(decx::div_m_fvec8_ST,
                                                A + (i << 3) * t_arrange_info._prev_proc_len,
                                                B + (i << 3) * t_arrange_info._prev_proc_len,
                                                dst + (i << 3) * t_arrange_info._prev_proc_len,
                                                t_arrange_info._prev_proc_len);
        }
    }
    else {
        for (int i = 0; i < thread_num - 1; ++i) {
            __async_stream[i] = 
                decx::thread_pool.register_task(decx::div_m_fvec8_ST,
                                                A + (i << 3) * t_arrange_info._prev_proc_len,
                                                B + (i << 3) * t_arrange_info._prev_proc_len,
                                                dst + (i << 3) * t_arrange_info._prev_proc_len,
                                                t_arrange_info._prev_proc_len);
        }
        __async_stream[thread_num - 1] = 
            decx::thread_pool.register_task(decx::div_m_fvec8_ST,
                                            A + (t_arrange_info._prev_proc_len << 3),
                                            B + (t_arrange_info._prev_proc_len << 3),
                                            dst + (t_arrange_info._prev_proc_len << 3),
                                            t_arrange_info._leftover);
    }

    for (int i = 0; i < thread_num; ++i) {
        __async_stream[i].get();
    }

    delete[] __async_stream;
}


void decx::Kdiv_m(int* A, int* B, int* dst, const size_t len)
{
    const uint thread_num = decx::cpI.cpu_concurrency;

    decx::utils::_thr_1D t_arrange_info(thread_num, len / 8);

    std::future<void>* __async_stream = new std::future<void>[thread_num];

    if (t_arrange_info.is_avg) {
        for (int i = 0; i < thread_num; ++i) {
            __async_stream[i] =
                decx::thread_pool.register_task(decx::div_m_ivec8_ST,
                    (__m256i*)A + i * t_arrange_info._prev_proc_len,
                    (__m256i*)B + i * t_arrange_info._prev_proc_len,
                    (__m256i*)dst + i * t_arrange_info._prev_proc_len,
                    t_arrange_info._prev_proc_len);
        }
    }
    else {
        for (int i = 0; i < thread_num - 1; ++i) {
            __async_stream[i] =
                decx::thread_pool.register_task(decx::div_m_ivec8_ST,
                    (__m256i*)A + i * t_arrange_info._prev_proc_len,
                    (__m256i*)B + i * t_arrange_info._prev_proc_len,
                    (__m256i*)dst + i * t_arrange_info._prev_proc_len,
                    t_arrange_info._prev_proc_len);
        }
        __async_stream[thread_num - 1] =
            decx::thread_pool.register_task(decx::div_m_ivec8_ST,
                (__m256i*)A + t_arrange_info._prev_proc_len,
                (__m256i*)B + t_arrange_info._prev_proc_len,
                (__m256i*)dst + t_arrange_info._prev_proc_len,
                t_arrange_info._leftover);
    }

    for (int i = 0; i < thread_num; ++i) {
        __async_stream[i].get();
    }

    delete[] __async_stream;
}



void decx::Kdiv_m(double* A, double* B, double* dst, const size_t len)
{
    const uint thread_num = decx::cpI.cpu_concurrency;

    decx::utils::_thr_1D t_arrange_info(thread_num, len / 4);

    std::future<void>* __async_stream = new std::future<void>[thread_num];

    if (t_arrange_info.is_avg) {
        for (int i = 0; i < thread_num; ++i) {
            __async_stream[i] =
                decx::thread_pool.register_task(decx::div_m_dvec4_ST,
                    A + (i << 2) * t_arrange_info._prev_proc_len,
                    B + (i << 2) * t_arrange_info._prev_proc_len,
                    dst + (i << 2) * t_arrange_info._prev_proc_len,
                    t_arrange_info._prev_proc_len);
        }
    }
    else {
        for (int i = 0; i < thread_num - 1; ++i) {
            __async_stream[i] =
                decx::thread_pool.register_task(decx::div_m_dvec4_ST,
                    A + (i << 2) * t_arrange_info._prev_proc_len,
                    B + (i << 2) * t_arrange_info._prev_proc_len,
                    dst + (i << 2) * t_arrange_info._prev_proc_len,
                    t_arrange_info._prev_proc_len);
        }
        __async_stream[thread_num - 1] =
            decx::thread_pool.register_task(decx::div_m_dvec4_ST,
                A + (t_arrange_info._prev_proc_len << 2),
                B + (t_arrange_info._prev_proc_len << 2),
                dst + (t_arrange_info._prev_proc_len << 2),
                t_arrange_info._leftover);
    }

    for (int i = 0; i < thread_num; ++i) {
        __async_stream[i].get();
    }

    delete[] __async_stream;
}


// ------------------------------------- constant -------------------------------------------------------


void decx::Kdiv_c(float* src, const float __x, float* dst, const size_t len)
{
    const uint thread_num = decx::cpI.cpu_concurrency;

    decx::utils::_thr_1D t_arrange_info(thread_num, len / 8);

    std::future<void>* __async_stream = new std::future<void>[thread_num];

    if (t_arrange_info.is_avg) {
        for (int i = 0; i < thread_num; ++i) {
            __async_stream[i] =
                decx::thread_pool.register_task(decx::div_c_fvec8_ST,
                    src + (i << 3) * t_arrange_info._prev_proc_len,
                    __x,
                    dst + (i << 3) * t_arrange_info._prev_proc_len,
                    t_arrange_info._prev_proc_len);
        }
    }
    else {
        for (int i = 0; i < thread_num - 1; ++i) {
            __async_stream[i] =
                decx::thread_pool.register_task(decx::div_c_fvec8_ST,
                    src + (i << 3) * t_arrange_info._prev_proc_len,
                    __x,
                    dst + (i << 3) * t_arrange_info._prev_proc_len,
                    t_arrange_info._prev_proc_len);
        }
        __async_stream[thread_num - 1] =
            decx::thread_pool.register_task(decx::div_c_fvec8_ST,
                src + (t_arrange_info._prev_proc_len << 3),
                __x,
                dst + (t_arrange_info._prev_proc_len << 3),
                t_arrange_info._leftover);
    }

    for (int i = 0; i < thread_num; ++i) {
        __async_stream[i].get();
    }

    delete[] __async_stream;
}


void decx::Kdiv_c(int* src, const int __x, int* dst, const size_t len)
{
    const uint thread_num = decx::cpI.cpu_concurrency;

    decx::utils::_thr_1D t_arrange_info(thread_num, len / 8);

    std::future<void>* __async_stream = new std::future<void>[thread_num];

    if (t_arrange_info.is_avg) {
        for (int i = 0; i < thread_num; ++i) {
            __async_stream[i] =
                decx::thread_pool.register_task(decx::div_c_ivec8_ST,
                    (__m256i*)src + i * t_arrange_info._prev_proc_len,
                    __x,
                    (__m256i*)dst + i * t_arrange_info._prev_proc_len,
                    t_arrange_info._prev_proc_len);
        }
    }
    else {
        for (int i = 0; i < thread_num - 1; ++i) {
            __async_stream[i] =
                decx::thread_pool.register_task(decx::div_c_ivec8_ST,
                    (__m256i*)src + i * t_arrange_info._prev_proc_len,
                    __x,
                    (__m256i*)dst + i * t_arrange_info._prev_proc_len,
                    t_arrange_info._prev_proc_len);
        }
        __async_stream[thread_num - 1] =
            decx::thread_pool.register_task(decx::div_c_ivec8_ST,
                (__m256i*)src + t_arrange_info._prev_proc_len,
                __x,
                (__m256i*)dst + t_arrange_info._prev_proc_len,
                t_arrange_info._leftover);
    }

    for (int i = 0; i < thread_num; ++i) {
        __async_stream[i].get();
    }

    delete[] __async_stream;
}


void decx::Kdiv_c(double* src, const double __x, double* dst, const size_t len)
{
    const uint thread_num = decx::cpI.cpu_concurrency;

    decx::utils::_thr_1D t_arrange_info(thread_num, len / 4);

    std::future<void>* __async_stream = new std::future<void>[thread_num];

    if (t_arrange_info.is_avg) {
        for (int i = 0; i < thread_num; ++i) {
            __async_stream[i] =
                decx::thread_pool.register_task(decx::div_c_dvec4_ST,
                    src + (i << 2) * t_arrange_info._prev_proc_len,
                    __x,
                    dst + (i << 2) * t_arrange_info._prev_proc_len,
                    t_arrange_info._prev_proc_len);
        }
    }
    else {
        for (int i = 0; i < thread_num - 1; ++i) {
            __async_stream[i] =
                decx::thread_pool.register_task(decx::div_c_dvec4_ST,
                    src + (i << 2) * t_arrange_info._prev_proc_len,
                    __x,
                    dst + (i << 2) * t_arrange_info._prev_proc_len,
                    t_arrange_info._prev_proc_len);
        }
        __async_stream[thread_num - 1] =
            decx::thread_pool.register_task(decx::div_c_dvec4_ST,
                src + (t_arrange_info._prev_proc_len << 2),
                __x,
                dst + (t_arrange_info._prev_proc_len << 2),
                t_arrange_info._leftover);
    }

    for (int i = 0; i < thread_num; ++i) {
        __async_stream[i].get();
    }

    delete[] __async_stream;
}



void decx::Kdiv_cinv(float* src, const float __x, float* dst, const size_t len)
{
    const uint thread_num = decx::cpI.cpu_concurrency;

    decx::utils::_thr_1D t_arrange_info(thread_num, len / 8);

    std::future<void>* __async_stream = new std::future<void>[thread_num];

    if (t_arrange_info.is_avg) {
        for (int i = 0; i < thread_num; ++i) {
            __async_stream[i] =
                decx::thread_pool.register_task(decx::div_cinv_fvec8_ST,
                    src + (i << 3) * t_arrange_info._prev_proc_len,
                    __x,
                    dst + (i << 3) * t_arrange_info._prev_proc_len,
                    t_arrange_info._prev_proc_len);
        }
    }
    else {
        for (int i = 0; i < thread_num - 1; ++i) {
            __async_stream[i] =
                decx::thread_pool.register_task(decx::div_cinv_fvec8_ST,
                    src + (i << 3) * t_arrange_info._prev_proc_len,
                    __x,
                    dst + (i << 3) * t_arrange_info._prev_proc_len,
                    t_arrange_info._prev_proc_len);
        }
        __async_stream[thread_num - 1] =
            decx::thread_pool.register_task(decx::div_cinv_fvec8_ST,
                src + (t_arrange_info._prev_proc_len << 3),
                __x,
                dst + (t_arrange_info._prev_proc_len << 3),
                t_arrange_info._leftover);
    }

    for (int i = 0; i < thread_num; ++i) {
        __async_stream[i].get();
    }

    delete[] __async_stream;
}


void decx::Kdiv_cinv(int* src, const int __x, int* dst, const size_t len)
{
    const uint thread_num = decx::cpI.cpu_concurrency;

    decx::utils::_thr_1D t_arrange_info(thread_num, len / 8);

    std::future<void>* __async_stream = new std::future<void>[thread_num];

    if (t_arrange_info.is_avg) {
        for (int i = 0; i < thread_num; ++i) {
            __async_stream[i] =
                decx::thread_pool.register_task(decx::div_cinv_ivec8_ST,
                    (__m256i*)src + i * t_arrange_info._prev_proc_len,
                    __x,
                    (__m256i*)dst + i * t_arrange_info._prev_proc_len,
                    t_arrange_info._prev_proc_len);
        }
    }
    else {
        for (int i = 0; i < thread_num - 1; ++i) {
            __async_stream[i] =
                decx::thread_pool.register_task(decx::div_cinv_ivec8_ST,
                    (__m256i*)src + i * t_arrange_info._prev_proc_len,
                    __x,
                    (__m256i*)dst + i * t_arrange_info._prev_proc_len,
                    t_arrange_info._prev_proc_len);
        }
        __async_stream[thread_num - 1] =
            decx::thread_pool.register_task(decx::div_cinv_ivec8_ST,
                (__m256i*)src + t_arrange_info._prev_proc_len,
                __x,
                (__m256i*)dst + t_arrange_info._prev_proc_len,
                t_arrange_info._leftover);
    }

    for (int i = 0; i < thread_num; ++i) {
        __async_stream[i].get();
    }

    delete[] __async_stream;
}


void decx::Kdiv_cinv(double* src, const double __x, double* dst, const size_t len)
{
    const uint thread_num = decx::cpI.cpu_concurrency;

    decx::utils::_thr_1D t_arrange_info(thread_num, len / 4);

    std::future<void>* __async_stream = new std::future<void>[thread_num];

    if (t_arrange_info.is_avg) {
        for (int i = 0; i < thread_num; ++i) {
            __async_stream[i] =
                decx::thread_pool.register_task(decx::div_cinv_dvec4_ST,
                    src + (i << 2) * t_arrange_info._prev_proc_len,
                    __x,
                    dst + (i << 2) * t_arrange_info._prev_proc_len,
                    t_arrange_info._prev_proc_len);
        }
    }
    else {
        for (int i = 0; i < thread_num - 1; ++i) {
            __async_stream[i] =
                decx::thread_pool.register_task(decx::div_cinv_dvec4_ST,
                    src + (i << 2) * t_arrange_info._prev_proc_len,
                    __x,
                    dst + (i << 2) * t_arrange_info._prev_proc_len,
                    t_arrange_info._prev_proc_len);
        }
        __async_stream[thread_num - 1] =
            decx::thread_pool.register_task(decx::div_cinv_dvec4_ST,
                src + (t_arrange_info._prev_proc_len << 2),
                __x,
                dst + (t_arrange_info._prev_proc_len << 2),
                t_arrange_info._leftover);
    }

    for (int i = 0; i < thread_num; ++i) {
        __async_stream[i].get();
    }

    delete[] __async_stream;
}


#endif