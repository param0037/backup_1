/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#include "../../core/basic.h"


/**
* @brief : __dst[0] = __x1[0] * __x2  (x, y)
* @param __x1 : In type of float2
* @param __x2 : In type of de::CPf
* @param __dst : In type of float2
*/
#define C_mul_C_fp32(__x1, __x2, __dst){    \
    __dst.x = __fmul_rn(__x1.x, __x2.real)  - __fmul_rn(__x1.y, __x2.image);    \
    __dst.y = __fmul_rn(__x1.x, __x2.image) + __fmul_rn(__x1.y, __x2.real);     \
}


/**
* @brief : __dst[0] = __x1[0] * __x2  (x, y)
*          __dst[1] = __x1[1] * __x2  (z, w)
* @param __x1 : In type of float4
* @param __x2 : In type of de::CPf
*/
#define C_mul_C2_fp32(__x1, __x2, __dst){    \
    __dst.x = __fmul_rn(__x1.x, __x2.real)  - __fmul_rn(__x1.y, __x2.image);    \
    __dst.y = __fmul_rn(__x1.x, __x2.image) + __fmul_rn(__x1.y, __x2.real);     \
    __dst.z = __fmul_rn(__x1.z, __x2.real)  - __fmul_rn(__x1.w, __x2.image);    \
    __dst.w = __fmul_rn(__x1.z, __x2.image) + __fmul_rn(__x1.w, __x2.real);     \
}



/**
* @brief : __dst[0] = __x1[0] + __x2[0]  (x, y)
* @param __x1 : In type of float2
* @param __x2 : In type of float2
*/
#define C_ADD_C(__x1, __x2, __dst){  \
    __dst.x = __fadd_rn(__x1.x, __x2.x);  \
    __dst.y = __fadd_rn(__x1.y, __x2.y);  \
}



/**
* @brief : __dst[0] = __x1[0] + __x2[0]  (x, y)
* @param __x1 : In type of float2
* @param __x2 : In type of float2
*/
#define C_SUB_C(__x1, __x2, __dst){  \
    __dst.x = __fsub_rn(__x1.x, __x2.x);  \
    __dst.y = __fsub_rn(__x1.y, __x2.y);  \
}



/**
* @brief : __dst[0] = __x1[0] + __x2[0]  (x, y)
*          __dst[1] = __x1[1] + __x2[1]  (z, w)
* @param __x1 : In type of float4
* @param __x2 : In type of float4
*/
#define C2_ADD_C2(__x1, __x2, __dst){  \
    __dst.x = __fadd_rn(__x1.x, __x2.x);  \
    __dst.y = __fadd_rn(__x1.y, __x2.y);  \
    __dst.z = __fadd_rn(__x1.z, __x2.z);  \
    __dst.w = __fadd_rn(__x1.w, __x2.w);  \
}


/**
* @brief : __dst[0] = __x1[0] - __x2[0]  (x, y)
*          __dst[1] = __x1[1] - __x2[1]  (z, w)
* @param __x1 : In type of float4
* @param __x2 : In type of float4
*/
#define C2_SUB_C2(__x1, __x2, __dst){  \
    __dst.x = __fsub_rn(__x1.x, __x2.x);  \
    __dst.y = __fsub_rn(__x1.y, __x2.y);  \
    __dst.z = __fsub_rn(__x1.z, __x2.z);  \
    __dst.w = __fsub_rn(__x1.w, __x2.w);  \
}



#define C_SUM5(__ele){   \
    tmp.__ele = __fadd_rn(CmulC_buffer[0].__ele, CmulC_buffer[1].__ele);     \
    tmp.__ele = __fadd_rn(CmulC_buffer[2].__ele, tmp.__ele);       \
    tmp.__ele = __fadd_rn(CmulC_buffer[3].__ele, tmp.__ele);       \
    tmp.__ele = __fadd_rn(CmulC_buffer[4].__ele, tmp.__ele);       \
}


#define C_SUM4(__ele){   \
    tmp.__ele = __fadd_rn(CmulC_buffer[0].__ele, CmulC_buffer[1].__ele);     \
    tmp.__ele = __fadd_rn(CmulC_buffer[2].__ele, tmp.__ele);       \
    tmp.__ele = __fadd_rn(CmulC_buffer[3].__ele, tmp.__ele);       \
}


#define C_SUM3(_buffer, __ele, _opobj){   \
    _opobj.__ele = __fadd_rn(_buffer[0].__ele, _buffer[1].__ele);     \
    _opobj.__ele = __fadd_rn(_buffer[2].__ele, _opobj.__ele);       \
}


__device__
// ������ԭ�����
void C_mul_C_f_ang(de::complex_f* __C1, de::complex_f* __C2, de::complex_f* dst)
{
    dst->real = __C1->real * __C2->real;
    dst->image = __C1->image + __C2->image;;
}


__device__
void ang_2_xy(de::CPf* src, de::CPf* dst)
{
    float amp = src->real;
    dst->real = __cosf(src->image);
    dst->image = __sinf(src->image);
}


__device__
void R_mul_C_f(float __C1, de::complex_f* __C2, de::complex_f* dst)
{
    dst->real = __C2->real * __C1;
    dst->image = __C2->image * __C1;
}


__device__
// this operation is in-place
void inv_C_f(de::complex_f* dst)
{
    dst->real *= -1;
    dst->image *= -1;
}



__device__
void constructW_f(const float angle, de::CPf* dst)
{
    dst->real = __cosf(angle);
    dst->image = __sinf(angle);
}




#define C_mul_real(_C, _real, _imag) (_C->real * (_real) - _C->image * (_imag))
#define C_mul_imag(_C, _real, _imag) (_C->real * (_imag) + _C->image * (_real))



#define _C_mul_real(_C, _real, _imag) ((_C).real * (_real) - (_C).image * (_imag))
#define _C_mul_imag(_C, _real, _imag) ((_C).real * (_imag) + (_C).image * (_real))



#define C_FMA_C_fp32_constant(__x1, _real, _image, _dst){           \
    _dst.x = __fadd_rn(__fmul_rn(__x1.x, (_real)), _dst.x);         \
    _dst.x = __fsub_rn(_dst.x, __fmul_rn(__x1.y, (_image)));        \
    _dst.y = __fadd_rn(__fmul_rn(__x1.x, (_image)), _dst.y);        \
    _dst.y = __fadd_rn(__fmul_rn(__x1.y, (_real)), _dst.y);         \
}



#define C_FMA_C_fp32_constant_Ignore_Image(__x1, _real, _image, _dst){           \
    _dst.x = __fadd_rn(__fmul_rn(__x1.x, (_real)), _dst.x);         \
    _dst.x = __fsub_rn(_dst.x, __fmul_rn(__x1.y, (_image)));        \
}



#define C_FMA_C2_fp32_constant(__x1, _real, _image, _dst){          \
    _dst.x = __fadd_rn(__fmul_rn(__x1.x, (_real)), _dst.x);         \
    _dst.x = __fsub_rn(_dst.x, __fmul_rn(__x1.y, (_image)));        \
    _dst.y = __fadd_rn(__fmul_rn(__x1.x, (_image)), _dst.y);        \
    _dst.y = __fadd_rn(__fmul_rn(__x1.y, (_real)), _dst.y);         \
    _dst.z = __fadd_rn(__fmul_rn(__x1.z, (_real)), _dst.z);         \
    _dst.z = __fsub_rn(_dst.z, __fmul_rn(__x1.w, (_image)));        \
    _dst.w = __fadd_rn(__fmul_rn(__x1.z, (_image)), _dst.w);        \
    _dst.w = __fadd_rn(__fmul_rn(__x1.w, (_real)), _dst.w);         \
}



#define R_mul_real(_R, _real, _imag) (*_R * (_real))
#define R_mul_imag(_R, _real, _imag) (*_R * (_imag))


#define R_mul_real_NPTR(_R, _real, _imag) (_R * (_real))
#define R_mul_imag_NPTR(_R, _real, _imag) (_R * (_imag))
