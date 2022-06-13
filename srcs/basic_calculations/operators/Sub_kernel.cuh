/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

/* /////////////////////////////////////////////////////////////////////////////
* The _2D kernel function is designed for Tensor and TensorArray, treat Tensor
* and TensorArray as a 2D matrix, with eq_pitch = ~.dpitch * ~.width, height of
* ~.height (Tensor) or ~.tensor_num * ~.height (TensorArray)
*/

#ifndef _SUB_KERNEL_CUH_
#define _SUB_KERNEL_CUH_

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
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void sub_m_ivec4_2D(float4* A, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    int4 tmpA, tmpB, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpA) = A[dex];
        *((float4*)&tmpB) = B[dex];

        tmpdst.x = tmpA.x - tmpB.x;
        tmpdst.y = tmpA.y - tmpB.y;
        tmpdst.z = tmpA.z - tmpB.z;
        tmpdst.w = tmpA.w - tmpB.w;

        dst[dex] = *((float4*)&tmpdst);
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
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void sub_m_fvec4_2D(float4* A, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    float4 tmpA, tmpB, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        tmpA = A[dex];
        tmpB = B[dex];

        tmpdst.x = __fsub_rn(tmpA.x, tmpB.x);
        tmpdst.y = __fsub_rn(tmpA.y, tmpB.y);
        tmpdst.z = __fsub_rn(tmpA.z, tmpB.z);
        tmpdst.w = __fsub_rn(tmpA.w, tmpB.w);

        dst[dex] = tmpdst;
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
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void sub_m_hvec8_2D(float4* A, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
#if __ABOVE_SM_53
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    half2_8 tmpA, tmpB, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpA) = A[dex];
        *((float4*)&tmpB) = B[dex];

        tmpdst.x = __hsub2(tmpA.x, tmpB.x);
        tmpdst.y = __hsub2(tmpA.y, tmpB.y);
        tmpdst.z = __hsub2(tmpA.z, tmpB.z);
        tmpdst.w = __hsub2(tmpA.w, tmpB.w);

        dst[dex] = *((float4*)&tmpdst);
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



__global__
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void sub_m_dvec2_2D(float4* A, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    double2 tmpA, tmpB, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpA) = A[dex];
        *((float4*)&tmpB) = B[dex];

        tmpdst.x = __dsub_rn(tmpA.x, tmpB.x);
        tmpdst.y = __dsub_rn(tmpA.y, tmpB.y);

        dst[dex] = *((float4*)&tmpdst);
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
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void sub_c_ivec4_2D(float4* src, int __x, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    int4 tmpsrc, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpsrc) = src[dex];

        tmpdst.x = tmpsrc.x - __x;
        tmpdst.y = tmpsrc.y - __x;
        tmpdst.z = tmpsrc.z - __x;
        tmpdst.w = tmpsrc.w - __x;

        dst[dex] = *((float4*)&tmpdst);
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
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void sub_cinv_ivec4_2D(float4* src, int __x, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    int4 tmpsrc, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpsrc) = src[dex];

        tmpdst.x = __x - tmpsrc.x;
        tmpdst.y = __x - tmpsrc.y;
        tmpdst.z = __x - tmpsrc.z;
        tmpdst.w = __x - tmpsrc.w;

        dst[dex] = *((float4*)&tmpdst);
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
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void sub_c_fvec4_2D(float4* src, float __x, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    float4 tmpsrc, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        tmpsrc = src[dex];

        tmpdst.x = __fsub_rn(tmpsrc.x, __x);
        tmpdst.y = __fsub_rn(tmpsrc.y, __x);
        tmpdst.z = __fsub_rn(tmpsrc.z, __x);
        tmpdst.w = __fsub_rn(tmpsrc.w, __x);

        dst[dex] = tmpdst;
    }
}



__global__
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void sub_cinv_fvec4_2D(float4* src, float __x, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    float4 tmpsrc, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        tmpsrc = src[dex];

        tmpdst.x = __fsub_rn(__x, tmpsrc.x);
        tmpdst.y = __fsub_rn(__x, tmpsrc.y);
        tmpdst.z = __fsub_rn(__x, tmpsrc.z);
        tmpdst.w = __fsub_rn(__x, tmpsrc.w);

        dst[dex] = tmpdst;
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
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void sub_c_hvec8_2D(float4* src, half2 __x, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
#if __ABOVE_SM_53
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    half2_8 tmpsrc, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpsrc) = src[dex];

        tmpdst.x = __hsub2(tmpsrc.x, __x);
        tmpdst.y = __hsub2(tmpsrc.y, __x);
        tmpdst.z = __hsub2(tmpsrc.z, __x);
        tmpdst.w = __hsub2(tmpsrc.w, __x);

        dst[dex] = *((float4*)&tmpdst);
    }
#endif
}



__global__
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void sub_cinv_hvec8_2D(float4* src, half2 __x, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
#if __ABOVE_SM_53
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    half2_8 tmpsrc, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpsrc) = src[dex];

        tmpdst.x = __hsub2(__x, tmpsrc.x);
        tmpdst.y = __hsub2(__x, tmpsrc.y);
        tmpdst.z = __hsub2(__x, tmpsrc.z);
        tmpdst.w = __hsub2(__x, tmpsrc.w);

        dst[dex] = *((float4*)&tmpdst);
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



__global__
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void sub_c_dvec2_2D(float4* src, double __x, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    double2 tmpsrc, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpsrc) = src[dex];

        tmpdst.x = __dsub_rn(tmpsrc.x, __x);
        tmpdst.y = __dsub_rn(tmpsrc.y, __x);

        dst[dex] = *((float4*)&tmpdst);
    }
}



__global__
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void sub_cinv_dvec2_2D(float4* src, double __x, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    double2 tmpsrc, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpsrc) = src[dex];

        tmpdst.x = __dsub_rn(__x, tmpsrc.x);
        tmpdst.y = __dsub_rn(__x, tmpsrc.y);

        dst[dex] = *((float4*)&tmpdst);
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
    // float
    static void dev_Ksub_m(float* DA, float* DB, float* Ddst, const size_t len) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);
        
        sub_m_fvec4 << < decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _DAptr, _DBptr, _Ddstptr, len / 4);
    }

    static void dev_Ksub_m_2D(float* DA, float* DB, float* Ddst, const size_t eq_pitch, const uint2 bounds) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        dim3 block(16, 16);
        dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

        sub_m_fvec4_2D << <grid, block >> > (_DAptr, _DBptr, _Ddstptr, eq_pitch / 4, bounds);
    }
    // end float

    // int
    static void dev_Ksub_m(int* DA, int* DB, int* Ddst, const size_t len) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        sub_m_ivec4 << < decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _DAptr, _DBptr, _Ddstptr, len / 4);
    }

    static void dev_Ksub_m_2D(int* DA, int* DB, int* Ddst, const size_t eq_pitch, const uint2 bounds) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        dim3 block(16, 16);
        dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

        sub_m_ivec4_2D << <grid, block >> > (_DAptr, _DBptr, _Ddstptr, eq_pitch / 4, bounds);
    }
    // end int

    // de::Half
    static void dev_Ksub_m(de::Half* DA, de::Half* DB, de::Half* Ddst, const size_t len) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        sub_m_hvec8 << < decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _DAptr, _DBptr, _Ddstptr, len / 8);
    }

    static void dev_Ksub_m_2D(de::Half* DA, de::Half* DB, de::Half* Ddst, const size_t eq_pitch, const uint2 bounds) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        dim3 block(16, 16);
        dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 8, 16));

        sub_m_hvec8_2D << <grid, block >> > (_DAptr, _DBptr, _Ddstptr, eq_pitch / 8, bounds);
    }
    // end de::Half

    // double
    static void dev_Ksub_m(double* DA, double* DB, double* Ddst, const size_t len) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        sub_m_dvec2 << < decx::utils::ceil<size_t>(len / 2, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
            _DAptr, _DBptr, _Ddstptr, len / 2);
    }

    static void dev_Ksub_m_2D(double* DA, double* DB, double* Ddst, const size_t eq_pitch, const uint2 bounds) {
        float4* _DAptr = reinterpret_cast<float4*>(DA);
        float4* _DBptr = reinterpret_cast<float4*>(DB);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        dim3 block(16, 16);
        dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 2, 16));

        sub_m_dvec2_2D << <grid, block >> > (_DAptr, _DBptr, _Ddstptr, eq_pitch / 2, bounds);
    }
    // end double

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

    static void dev_Ksub_c_2D(float* Dsrc, float __x, float* Ddst, const size_t eq_pitch, const uint2 bounds) {
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        dim3 block(16, 16);
        dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

        sub_c_fvec4_2D << <grid, block >> > (_Dsrcptr, __x, _Ddstptr, eq_pitch / 4, bounds);
    }

    static void dev_Ksub_cinv_2D(float* Dsrc, float __x, float* Ddst, const size_t eq_pitch, const uint2 bounds) {
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        dim3 block(16, 16);
        dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

        sub_cinv_fvec4_2D << <grid, block >> > (_Dsrcptr, __x, _Ddstptr, eq_pitch / 4, bounds);
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

    static void dev_Ksub_c_2D(int* Dsrc, int __x, int* Ddst, const size_t eq_pitch, const uint2 bounds) {
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        dim3 block(16, 16);
        dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

        sub_c_ivec4_2D << <grid, block >> > (_Dsrcptr, __x, _Ddstptr, eq_pitch / 4, bounds);
    }

    static void dev_Ksub_cinv_2D(int* Dsrc, int __x, int* Ddst, const size_t eq_pitch, const uint2 bounds) {
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        dim3 block(16, 16);
        dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

        sub_cinv_ivec4_2D << <grid, block >> > (_Dsrcptr, __x, _Ddstptr, eq_pitch / 4, bounds);
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

    static void dev_Ksub_c_2D(de::Half* Dsrc, de::Half __x, de::Half* Ddst, const size_t eq_pitch, const uint2 bounds) {
        half2 _x;
        _x.x = *((__half*)&__x);                _x.y = *((__half*)&__x);
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        dim3 block(16, 16);
        dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 8, 16));

        sub_c_hvec8_2D << <grid, block >> > (_Dsrcptr, _x, _Ddstptr, eq_pitch / 8, bounds);
    }

    static void dev_Ksub_cinv_2D(de::Half* Dsrc, de::Half __x, de::Half* Ddst, const size_t eq_pitch, const uint2 bounds) {
        half2 _x;
        _x.x = *((__half*)&__x);                _x.y = *((__half*)&__x);
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        dim3 block(16, 16);
        dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 8, 16));

        sub_cinv_hvec8_2D << <grid, block >> > (_Dsrcptr, _x, _Ddstptr, eq_pitch / 8, bounds);
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

    static void dev_Ksub_c_2D(double* Dsrc, double __x, double* Ddst, const size_t eq_pitch, const uint2 bounds) {
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        dim3 block(16, 16);
        dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 2, 16));

        sub_c_dvec2_2D << <grid, block >> > (_Dsrcptr, __x, _Ddstptr, eq_pitch / 2, bounds);
    }

    static void dev_Ksub_cinv_2D(double* Dsrc, double __x, double* Ddst, const size_t eq_pitch, const uint2 bounds) {
        float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
        float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

        dim3 block(16, 16);
        dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 2, 16));

        sub_cinv_dvec2_2D << <grid, block >> > (_Dsrcptr, __x, _Ddstptr, eq_pitch / 2, bounds);
    }
    // end double
}


#endif