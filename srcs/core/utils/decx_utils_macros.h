/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once


#define GetLarger(__a, __b) ((__a) > (__b) ? (__a) : (__b))
#define GetSmaller(__a, __b) ((__a) < (__b) ? (__a) : (__b))


#define SWAP(A, B, tmp) {    \
(tmp) = (A);    \
(A) = (B);    \
(B) = (tmp);    \
}


#define SWAP_(A, B, tmp) {    \
(A) = (tmp);    \
(B) = (tmp);    \
}


#define GetValue(src, _row, _col, cols) (src)[(_row) * (cols) + (_col)]

#ifndef GNU_CPUcodes
#define _MAX_TPB_(_prop, _device) cudaGetDeviceProperties(&_prop, _device);
#endif

#define __SPACE__ sizeof(T)


#define __TYPE__        de::Matrix<T>
#define _T_TYPE_        de::Tensor<T>
#define _VTYPE_         de::Vector<T>

#define _INT_            de::Matrix<int>
#define _FLOAT_            de::Matrix<float>
#define _DOUBLE_        de::Matrix<double>
#define _SHORT_            de::Matrix<short>
#define _UCHAR_            de::Matrix<uchar>


#define _VINT_          de::Vector<int>
#define _VFLOAT_        de::Vector<float>
#define _VDOUBLE_       de::Vector<double>
#define _VHALF_         de::Vector<de::Half>
#define _VCPF_          de::Vector<de::CPf>


#define _CPF_           de::Matrix<de::complex_f>
#ifndef GNU_CPUcodes
#define _CPH_           de::Matrix<de::complex_h>
#define _HALF_          de::Matrix<de::Half>
#endif

#define T_INT            de::Tensor<int>
#define T_FLOAT            de::Tensor<float>
#define T_DOUBLE        de::Tensor<double>
#define T_SHORT            de::Tensor<short>
#define T_UCHAR            de::Tensor<uchar>


#define T_CPF_          de::Tensor<de::complex_f>
#ifndef GNU_CPUcodes
#define T_CPH_          de::Tensor<de::complex_h>
#define T_HALF          de::Tensor<de::Half>
#endif





#ifdef Windows
#define _DECX_API_ __declspec(dllexport)
#endif

#ifdef Linux
#define _DECX_API_ __attribute__((visibility("default")))
#endif



// the hardware infomation
// the most blocks can execute concurrently in one SM
#define most_bl_per_sm 8



#ifndef __align__
#define __align__(n) __declspec(align(n))
#endif


typedef unsigned char uchar;
typedef unsigned int uint;