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
* 对于二字节的元素我用 vec8 类型访问全局内存，对于 x8 的列数，都可以整除，因此，核函数不用考虑边界
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
void fma_m_ivec4(float4* A, float4* B, float4 *C, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    int4 tmpA, tmpB, tmpC, tmpdst;

    if (tid < len) {
        tmpA = *((int4*)&A[tid]);
        tmpB = *((int4*)&B[tid]);
        tmpC = *((int4*)&C[tid]);

        tmpdst.x = tmpA.x * tmpB.x + tmpC.x;
        tmpdst.y = tmpA.y * tmpB.y + tmpC.y;
        tmpdst.z = tmpA.z * tmpB.z + tmpC.z;
        tmpdst.w = tmpA.w * tmpB.w + tmpC.w;

        dst[tid] = *((float4*)&tmpdst);
    }
}




__global__
/**
* int* x2, add together
* @param len : have considered vec4
*/
void fma_m_fvec4(float4* A, float4* B, float4* C, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 tmpA, tmpB, tmpC, tmpdst;

    if (tid < len) {
        tmpA = A[tid];
        tmpB = B[tid];
        tmpC = C[tid];

        tmpdst.x = fmaf(tmpA.x, tmpB.x, tmpC.x);
        tmpdst.y = fmaf(tmpA.y, tmpB.y, tmpC.y);
        tmpdst.z = fmaf(tmpA.z, tmpB.z, tmpC.z);
        tmpdst.w = fmaf(tmpA.w, tmpB.w, tmpC.w);

        dst[tid] = tmpdst;
    }
}





__global__
void fma_m_hvec8(float4* A, float4* B, float4* C, float4* dst, const size_t len)
{
#if __ABOVE_SM_53
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    half2_8 tmpA, tmpB, tmpC, tmpdst;

    if (tid < len) {
        tmpA = *((half2_8*)&A[tid]);
        tmpB = *((half2_8*)&B[tid]);
        tmpC = *((half2_8*)&C[tid]);

        tmpdst.x = __hfma2(tmpA.x, tmpB.x, tmpC.x);
        tmpdst.y = __hfma2(tmpA.y, tmpB.y, tmpC.y);
        tmpdst.z = __hfma2(tmpA.z, tmpB.z, tmpC.z);
        tmpdst.w = __hfma2(tmpA.w, tmpB.w, tmpC.w);

        dst[tid] = *((float4*)&tmpdst);
    }
#endif
}



__global__
/**
* int* x2, add together
* @param len : have considered vec4
*/
void fma_m_dvec2(float4* A, float4* B, float4* C, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    double2 tmpA, tmpB, tmpC, tmpdst;

    if (tid < len) {
        tmpA = *((double2*)&A[tid]);
        tmpB = *((double2*)&B[tid]);
        tmpC = *((double2*)&C[tid]);

        tmpdst.x = fma(tmpA.x, tmpB.x, tmpC.x);
        tmpdst.y = fma(tmpA.y, tmpB.y, tmpC.y);

        dst[tid] = *((float4*)&tmpdst);
    }
}



// ----------------------------- C --------------------------------------


__global__
void fma_c_ivec4(float4* A, int __x, float4 *B, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    int4 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = *((int4*)&A[tid]);
        tmpB = *((int4*)&B[tid]);

        tmpdst.x = tmpA.x * __x + tmpB.x;
        tmpdst.y = tmpA.y * __x + tmpB.y;
        tmpdst.z = tmpA.z * __x + tmpB.z;
        tmpdst.w = tmpA.w * __x + tmpB.w;

        dst[tid] = *((float4*)&tmpdst);
    }
}





__global__
void fma_c_fvec4(float4* A, float __x, float4 *B, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = A[tid];
        tmpB = B[tid];

        tmpdst.x = fmaf(tmpA.x, __x, tmpB.x);
        tmpdst.y = fmaf(tmpA.y, __x, tmpB.y);
        tmpdst.z = fmaf(tmpA.z, __x, tmpB.z);
        tmpdst.w = fmaf(tmpA.w, __x, tmpB.w);

        dst[tid] = tmpdst;
    }
}




__global__
void fma_c_hvec8(float4* A, half2 __x, float4 *B, float4* dst, const size_t len)
{
#if __ABOVE_SM_53
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    half2_8 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = *((half2_8*)&A[tid]);
        tmpB = *((half2_8*)&B[tid]);

        tmpdst.x = __hfma2(tmpA.x, __x, tmpB.x);
        tmpdst.y = __hfma2(tmpA.y, __x, tmpB.y);
        tmpdst.z = __hfma2(tmpA.z, __x, tmpB.z);
        tmpdst.w = __hfma2(tmpA.w, __x, tmpB.w);

        dst[tid] = *((float4*)&tmpdst);
    }
#endif
}



__global__
void fma_c_dvec2(float4* A, double __x, float4* B, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    double2 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = *((double2*)&A[tid]);
        tmpB = *((double2*)&B[tid]);

        tmpdst.x = fma(tmpA.x, __x, tmpB.x);
        tmpdst.y = fma(tmpA.y, __x, tmpB.y);

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
    static void dev_Kfma_m(float* DA, float* DB, float* DC, float* Ddst, const size_t len) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _DCptr = reinterpret_cast<float4*>(DC);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        fma_m_fvec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _DAptr, _DBptr, _DCptr, _Ddstptr, len / 4);
    }

    static void dev_Kfma_m(int* DA, int* DB, int* DC, int* Ddst, const size_t len) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _DCptr = reinterpret_cast<float4*>(DC);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        fma_m_ivec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _DAptr, _DBptr, _DCptr, _Ddstptr, len / 4);
    }


    static void dev_Kfma_m(de::Half* DA, de::Half* DB, de::Half* DC, de::Half* Ddst, const size_t len) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _DCptr = reinterpret_cast<float4*>(DC);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        fma_m_hvec8 << <decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _DAptr, _DBptr, _DCptr, _Ddstptr, len / 8);
    }

    
    static void dev_Kfma_m(double* DA, double* DB, double* DC, double* Ddst, const size_t len) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _DCptr = reinterpret_cast<float4*>(DC);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        fma_m_dvec2 << <decx::utils::ceil<size_t>(len / 2, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _DAptr, _DBptr, _DCptr, _Ddstptr, len / 2);
    }


    //// ----------------------------------------------- C -------------------------------------------------------------




    
    static void dev_Kfma_c(float* DA, float __x, float* DB, float* Ddst, const size_t len) {
        float4* Dptr_A = reinterpret_cast<float4*>(DA);
        float4* Dptr_B = reinterpret_cast<float4*>(DB);
        float4* Dptr_dst = reinterpret_cast<float4*>(Ddst);
        fma_c_fvec4 << < decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            Dptr_A, __x, Dptr_B, Dptr_dst, len / 4);
    }

    
    static void dev_Kfma_c(int* DA, int __x, int* DB, int* Ddst, const size_t len) {
        float4* Dptr_A = reinterpret_cast<float4*>(DA);
        float4* Dptr_B = reinterpret_cast<float4*>(DB);
        float4* Dptr_dst = reinterpret_cast<float4*>(Ddst);
        fma_c_ivec4 << < decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            Dptr_A, __x, Dptr_B, Dptr_dst, len / 4);
    }

    
    static void dev_Kfma_c(de::Half* DA, de::Half __x, de::Half* DB, de::Half* Ddst, const size_t len) {
        half2 _x;
        _x.x = *((__half*)&__x);                _x.y = *((__half*)&__x);
        float4* Dptr_A = reinterpret_cast<float4*>(DA);
        float4* Dptr_B = reinterpret_cast<float4*>(DB);
        float4* Dptr_dst = reinterpret_cast<float4*>(Ddst);
        fma_c_hvec8 << < decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            Dptr_A, _x, Dptr_B, Dptr_dst, len / 8);
    }

    
    static void dev_Kfma_c(double* DA, double __x, double* DB, double* Ddst, const size_t len) {
        float4* Dptr_A = reinterpret_cast<float4*>(DA);
        float4* Dptr_B = reinterpret_cast<float4*>(DB);
        float4* Dptr_dst = reinterpret_cast<float4*>(Ddst);
        fma_c_dvec2 << < decx::utils::ceil<size_t>(len / 2, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            Dptr_A, __x, Dptr_B, Dptr_dst, len / 2);
    }

}