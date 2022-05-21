/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.9.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.9.16
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
void add_m_ivec4(float4* A, float4* B, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    int4 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = *((int4*)&A[tid]);
        tmpB = *((int4*)&B[tid]);
        
        tmpdst.x = tmpA.x + tmpB.x;
        tmpdst.y = tmpA.y + tmpB.y;
        tmpdst.z = tmpA.z + tmpB.z;
        tmpdst.w = tmpA.w + tmpB.w;

        dst[tid] = *((float4*)&tmpdst);
    }
}




__global__
/**
* int* x2, add together
* @param len : have considered vec4
*/
void add_m_fvec4(float4* A, float4* B, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = A[tid];
        tmpB = B[tid];

        tmpdst.x = tmpA.x + tmpB.x;
        tmpdst.y = tmpA.y + tmpB.y;
        tmpdst.z = tmpA.z + tmpB.z;
        tmpdst.w = tmpA.w + tmpB.w;

        dst[tid] = tmpdst;
    }
}





__global__
void add_m_hvec8(float4* A, float4* B, float4* dst, const size_t len)
{
#if __ABOVE_SM_53
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    half2_8 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = *((half2_8*)&A[tid]);
        tmpB = *((half2_8*)&B[tid]);

        tmpdst.x = __hadd2(tmpA.x, tmpB.x);
        tmpdst.y = __hadd2(tmpA.y, tmpB.y);
        tmpdst.z = __hadd2(tmpA.z, tmpB.z);
        tmpdst.w = __hadd2(tmpA.w, tmpB.w);

        dst[tid] = *((float4*)&tmpdst);
    }
#endif
}



__global__
/**
* int* x2, add together
* @param len : have considered vec4
*/
void add_m_dvec2(float4* A, float4* B, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    double2 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = *((double2*)&A[tid]);
        tmpB = *((double2*)&B[tid]);

        tmpdst.x = tmpA.x + tmpB.x;
        tmpdst.y = tmpA.y + tmpB.y;

        dst[tid] = *((float4*)&tmpdst);
    }
}



// ----------------------------- C --------------------------------------


__global__
void add_c_ivec4(float4* src, int __x, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    int4 tmpsrc, tmpdst;

    if (tid < len) {
        tmpsrc = *((int4*)&src[tid]);
        
        tmpdst.x = __fadd_rn(tmpsrc.x, __x);
        tmpdst.y = __fadd_rn(tmpsrc.y, __x);
        tmpdst.z = __fadd_rn(tmpsrc.z, __x);
        tmpdst.w = __fadd_rn(tmpsrc.w, __x);

        dst[tid] = *((float4*)&tmpdst);
    }
}





__global__
void add_c_fvec4(float4* src, float __x, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 tmpsrc, tmpdst;

    if (tid < len) {
        tmpsrc = src[tid];

        tmpdst.x = __fadd_rn(tmpsrc.x, __x);
        tmpdst.y = __fadd_rn(tmpsrc.y, __x);
        tmpdst.z = __fadd_rn(tmpsrc.z, __x);
        tmpdst.w = __fadd_rn(tmpsrc.w, __x);

        dst[tid] = tmpdst;
    }
}




__global__
void add_c_hvec8(float4* src, half2 __x, float4* dst, const size_t len)
{
#if __ABOVE_SM_53
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    half2_8 tmpsrc, tmpdst;

    if (tid < len) {
        tmpsrc = *((half2_8*)&src[tid]);

        tmpdst.x = __hadd2(tmpsrc.x, __x);
        tmpdst.y = __hadd2(tmpsrc.y, __x);
        tmpdst.z = __hadd2(tmpsrc.z, __x);
        tmpdst.w = __hadd2(tmpsrc.w, __x);

        dst[tid] = *((float4*)&tmpdst);
    }
#endif
}



__global__
void add_c_dvec2(float4* src, double __x, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    double2 tmpsrc, tmpdst;

    if (tid < len) {
        tmpsrc = *((double2*)&src[tid]);

        tmpdst.x = __dadd_rn(tmpsrc.x, __x);
        tmpdst.y = __dadd_rn(tmpsrc.y, __x);

        dst[tid] = *((float4*)&tmpdst);
    }
}




// ---------------------------------------------------------------------------------------

// Sometime it is necessary for some single to be processed from host to device and back to host
// use zero stream (defualt stream avoiding the time cost on cudaDeviceSynchronize()
/** HOST float
* @param len : height * pitch, in the stride of original type, the function will divide it 
* by e.g. 4(sizeof(T) = 4). Because in kernel function, the load step is 128 bytes, to maximize the
* utilization of bandwidth and instruction cycle. Translated to load.128 in assembly.
*/


namespace decx
{
    
    static void dev_Kadd_m(float* DA, float* DB, float* Ddst, const size_t len) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);    
        float4* _DBptr = reinterpret_cast<float4*>(DB);    
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);    

        add_m_fvec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _DAptr, _DBptr, _Ddstptr, len / 4);
    }

    
    static void dev_Kadd_m(int* DA, int* DB, int* Ddst, const size_t len) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        add_m_ivec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _DAptr, _DBptr, _Ddstptr, len / 4);
    }

    

    static void dev_Kadd_m(de::Half* DA, de::Half* DB, de::Half* Ddst, const size_t len) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        add_m_hvec8 << <decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _DAptr, _DBptr, _Ddstptr, len / 8);
    }

    

    static void dev_Kadd_m(double* DA, double* DB, double* Ddst, const size_t len) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        add_m_dvec2 << <decx::utils::ceil<size_t>(len / 2, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _DAptr, _DBptr, _Ddstptr, len / 2);
    }


    // ----------------------------------------------- C -------------------------------------------------------------


    static void dev_Kadd_c(float* Dsrc, float __x, float* Ddst, const size_t len) {
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        add_c_fvec4 << < decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _Dsrcptr, __x, _Ddstptr, len / 4);
    }

    

    static void dev_Kadd_c(int* Dsrc, int __x, int* Ddst, const size_t len) {
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        add_c_ivec4 << < decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _Dsrcptr, __x, _Ddstptr, len / 4);
    }


    static void dev_Kadd_c(de::Half* Dsrc, de::Half __x, de::Half* Ddst, const size_t len) {
        half2 _x;
        _x.x = *((__half*)&__x);                _x.y = *((__half*)&__x);
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        add_c_hvec8 << < decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _Dsrcptr, _x, _Ddstptr, len / 8);
    }

    
    static void dev_Kadd_c(double* Dsrc, double __x, double* Ddst, const size_t len) {
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        add_c_dvec2 << < decx::utils::ceil<size_t>(len / 2, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _Dsrcptr, __x, _Ddstptr, len / 2);
    }

}