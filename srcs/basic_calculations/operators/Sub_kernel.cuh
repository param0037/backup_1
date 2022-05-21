/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

/* /////////////////////////////////////////////////////////////////////////////
* for the small matrix or tensor, the calculation assignment will be sent to CPU
* vectorize the memory access, 通常，对于四字节的元素我用 vec4 类型访问全局内存；
* 对于二字节的元素我用 vec8 类型访问全局内存，对于 x256 的列数，都可以整除，因此，核函数不用考虑边界
*/

#pragma once

#include "../../core/basic.h"
#include "../../classes/core_types.h"



// ------------------------- M ------------------------------------------

__global__
/**
* int* x2, add together
* @param len : have considered vec4
*/
void sub_m_ivec4(float4* A, float4* B, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    int4 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = *((int4*)&A[tid]);
        tmpB = *((int4*)&B[tid]);
        
        tmpdst.x = tmpA.x - tmpB.x;
        tmpdst.y = tmpA.y - tmpB.y;
        tmpdst.z = tmpA.z - tmpB.z;
        tmpdst.w = tmpA.w - tmpB.w;

        dst[tid] = *((float4*)&tmpdst);
    }
}




__global__
/**
* int* x2, add together
* @param len : have considered vec4
*/
void sub_m_fvec4(float4* A, float4* B, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = A[tid];
        tmpB = B[tid];

        tmpdst.x = __fsub_rn(tmpA.x, tmpB.x);
        tmpdst.y = __fsub_rn(tmpA.y, tmpB.y);
        tmpdst.z = __fsub_rn(tmpA.z, tmpB.z);
        tmpdst.w = __fsub_rn(tmpA.w, tmpB.w);

        dst[tid] = tmpdst;
    }
}




__global__
void sub_m_hvec8(float4* A, float4* B, float4* dst, const size_t len)
{
#if __ABOVE_SM_53
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    half2_8 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = *((half2_8*)&A[tid]);
        tmpB = *((half2_8*)&B[tid]);

        tmpdst.x = __hsub2(tmpA.x, tmpB.x);
        tmpdst.y = __hsub2(tmpA.y, tmpB.y);
        tmpdst.z = __hsub2(tmpA.z, tmpB.z);
        tmpdst.w = __hsub2(tmpA.w, tmpB.w);

        dst[tid] = *((float4*)&tmpdst);
    }
#endif
}



__global__
/**
* int* x2, add together
* @param len : have considered vec4
*/
void sub_m_dvec2(float4* A, float4* B, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    double2 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = *((double2*)&A[tid]);
        tmpB = *((double2*)&B[tid]);

        tmpdst.x = __dsub_rn(tmpA.x, tmpB.x);
        tmpdst.y = __dsub_rn(tmpA.y, tmpB.y);

        dst[tid] = *((float4*)&tmpdst);
    }
}



// ----------------------------- C --------------------------------------


__global__
void sub_c_ivec4(float4* src, int __x, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    int4 tmpsrc, tmpdst;

    if (tid < len) {
        tmpsrc = *((int4*)&src[tid]);
        
        tmpdst.x = tmpsrc.x - __x;
        tmpdst.y = tmpsrc.y - __x;
        tmpdst.z = tmpsrc.z - __x;
        tmpdst.w = tmpsrc.w - __x;

        dst[tid] = *((float4*)&tmpdst);
    }
}



__global__
void sub_cinv_ivec4(int __x, float4* src, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    int4 tmpsrc, tmpdst;

    if (tid < len) {
        tmpsrc = *((int4*)&src[tid]);

        tmpdst.x = __x - tmpsrc.x;
        tmpdst.y = __x - tmpsrc.y;
        tmpdst.z = __x - tmpsrc.z;
        tmpdst.w = __x - tmpsrc.w;

        dst[tid] = *((float4*)&tmpdst);
    }
}




__global__
void sub_c_fvec4(float4* src, float __x, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 tmpsrc, tmpdst;

    if (tid < len) {
        tmpsrc = src[tid];

        tmpdst.x = __fsub_rn(tmpsrc.x, __x);
        tmpdst.y = __fsub_rn(tmpsrc.y, __x);
        tmpdst.z = __fsub_rn(tmpsrc.z, __x);
        tmpdst.w = __fsub_rn(tmpsrc.w, __x);

        dst[tid] = tmpdst;
    }
}



__global__
void sub_cinv_fvec4(float __x, float4* src, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 tmpsrc, tmpdst;

    if (tid < len) {
        tmpsrc = src[tid];

        tmpdst.x = __fsub_rn(__x, tmpsrc.x);
        tmpdst.y = __fsub_rn(__x, tmpsrc.y);
        tmpdst.z = __fsub_rn(__x, tmpsrc.z);
        tmpdst.w = __fsub_rn(__x, tmpsrc.w);

        dst[tid] = tmpdst;
    }
}





__global__
void sub_c_hvec8(float4* src, half2 __x, float4* dst, const size_t len)
{
#if __ABOVE_SM_53
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    half2_8 tmpsrc, tmpdst;

    if (tid < len) {
        tmpsrc = *((half2_8*)&src[tid]);

        tmpdst.x = __hsub2(tmpsrc.x, __x);
        tmpdst.y = __hsub2(tmpsrc.y, __x);
        tmpdst.z = __hsub2(tmpsrc.z, __x);
        tmpdst.w = __hsub2(tmpsrc.w, __x);

        dst[tid] = *((float4*)&tmpdst);
    }
#endif
}



__global__
void sub_cinv_hvec8(half2 __x, float4* src, float4* dst, const size_t len)
{
#if __ABOVE_SM_53
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    half2_8 tmpsrc, tmpdst;

    if (tid < len) {
        tmpsrc = *((half2_8*)&src[tid]);

        tmpdst.x = __hsub2(__x, tmpsrc.x);
        tmpdst.y = __hsub2(__x, tmpsrc.y);
        tmpdst.z = __hsub2(__x, tmpsrc.z);
        tmpdst.w = __hsub2(__x, tmpsrc.w);

        dst[tid] = *((float4*)&tmpdst);
    }
#endif
}



__global__
void sub_c_dvec2(float4* src, double __x, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    double2 tmpsrc, tmpdst;

    if (tid < len) {
        tmpsrc = *((double2*)&src[tid]);

        tmpdst.x = __dsub_rn(tmpsrc.x, __x);
        tmpdst.y = __dsub_rn(tmpsrc.y, __x);
    
        dst[tid] = *((float4*)&tmpdst);
    }
}



__global__
void sub_cinv_dvec2(double __x, float4* src, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    double2 tmpsrc, tmpdst;

    if (tid < len) {
        tmpsrc = *((double2*)&src[tid]);

        tmpdst.x = __dsub_rn(__x, tmpsrc.x);
        tmpdst.y = __dsub_rn(__x, tmpsrc.y);

        dst[tid] = *((float4*)&tmpdst);
    }
}


// ---------------------------------------------------------------------------------------
// necessary single proccess from host to device and back to host
// use zero stream (defualt stream avoiding the time cost on cudaDeviceSynchronize()
/** HOST float
* @param len : height * pitch
*/


namespace decx
{
    static void dev_Ksub_m(float* DA, float* DB, float* Ddst, const size_t len) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);
        
        sub_m_fvec4 << < decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _DAptr, _DBptr, _Ddstptr, len / 4);
    }

    static void dev_Ksub_m(int* DA, int* DB, int* Ddst, const size_t len) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        sub_m_ivec4 << < decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _DAptr, _DBptr, _Ddstptr, len / 4);
    }


    static void dev_Ksub_m(de::Half* DA, de::Half* DB, de::Half* Ddst, const size_t len) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        sub_m_hvec8 << < decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _DAptr, _DBptr, _Ddstptr, len / 8);
    }


    static void dev_Ksub_m(double* DA, double* DB, double* Ddst, const size_t len) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        sub_m_dvec2 << < decx::utils::ceil<size_t>(len / 2, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _DAptr, _DBptr, _Ddstptr, len / 2);
    }


    // ----------------------------------------------- C -------------------------------------------------------------


    static void dev_Ksub_c(float* Dsrc, float __x, float* Ddst, const size_t len) {
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        sub_c_fvec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _Dsrcptr, __x, _Ddstptr, len / 4);
    }

    static void dev_Ksub_cinv(float __x, float* Dsrc, float* Ddst, const size_t len) {
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        sub_cinv_fvec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            __x, _Dsrcptr, _Ddstptr, len / 4);
    }
    // end float


    static void dev_Ksub_c(int* Dsrc, int __x, int* Ddst, const size_t len) {
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        sub_c_ivec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _Dsrcptr, __x, _Ddstptr, len / 4);
    }

    static void dev_Ksub_cinv(int __x, int* Dsrc, int* Ddst, const size_t len) {
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        sub_cinv_ivec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            __x, _Dsrcptr, _Ddstptr, len / 4);
    }
    // end int 

    // half
    
    static void dev_Ksub_c(de::Half* Dsrc, de::Half __x, de::Half* Ddst, const size_t len) {
        half2 _x;
        _x.x = *((__half*)&__x);                _x.y = *((__half*)&__x);
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        sub_c_hvec8 << <decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _Dsrcptr, _x, _Ddstptr, len / 8);
    }

    static void dev_Ksub_cinv(de::Half __x, de::Half* Dsrc, de::Half* Ddst, const size_t len) {
        half2 _x;
        _x.x = *((__half*)&__x);                _x.y = *((__half*)&__x);
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        sub_cinv_hvec8 << <decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _x, _Dsrcptr, _Ddstptr, len / 8);
    }
    // end half

    // double
    
    static void dev_Ksub_c(double* Dsrc, double __x, double* Ddst, const size_t len) {
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        sub_c_dvec2 << <decx::utils::ceil<size_t>(len / 2, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _Dsrcptr, __x, _Ddstptr, len / 2);
    }

    static void dev_Ksub_cinv(double __x, double* Dsrc, double* Ddst, const size_t len) {
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        sub_cinv_dvec2 << <decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            __x, _Dsrcptr, _Ddstptr, len / 8);
    }
    // end double

}