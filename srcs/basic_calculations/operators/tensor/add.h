/**
*	---------------------------------------------------------------------
*	Author : Wayne
*   Date   : 2021.9.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.9.16
*/

/**
* 处理思想： 只拷贝和处理有效的行数，但是列数必须全部处理
*		x256
* ---------------
* |*********....|
* |*********....|
* |*********....|	x128
* |*********....|
* |				|
* ---------------
* where "*" represents effective elements and "." represents idle elements
*/

#pragma once


#include "../Add_kernel.cuh"
#include "../../../basic/basic.h"


namespace de
{
	template <typename T>
	_DECX_API_  de::DH Add(_T_TYPE_& A, _T_TYPE_& B, _T_TYPE_& dst);



	template <typename T>
	_DECX_API_  de::DH Add(_T_TYPE_& src, T __x, _T_TYPE_& dst);
}



template <typename T>
de::DH de::Add(_T_TYPE_& A, _T_TYPE_& B, _T_TYPE_& dst)
{
	_Tensor<T>& _A = dynamic_cast<_Tensor<T>&>(A);
	_Tensor<T>& _B = dynamic_cast<_Tensor<T>&>(B);
	_Tensor<T>& _dst = dynamic_cast<_Tensor<T>&>(dst);

	de::DH handle;
	if (!cuP.is_init) {
		Not_init(&handle);
		return handle;
	}

	if (_A.width != _B.width || _A.height != _B.height) {
		MDim_Not_Matching(&handle);
		return handle;
	}

	uint alloc_d_width = _A.width * _A.pitch;
	Kadd_m(_A.Mat, _B.Mat, _dst.Mat, alloc_d_width, _A.height, alloc_d_width, alloc_d_width);

	Success(&handle);
	return handle;
}

template _DECX_API_ de::DH de::Add(T_FLOAT& A, T_FLOAT& B, T_FLOAT& dst);

template _DECX_API_ de::DH de::Add(T_INT& A, T_INT& B, T_INT& dst);

template _DECX_API_ de::DH de::Add(T_HALF& A, T_HALF& B, T_HALF& dst);

template _DECX_API_ de::DH de::Add(T_DOUBLE& A, T_DOUBLE& B, T_DOUBLE& dst);



template <typename T>
de::DH de::Add(_T_TYPE_& src, T __x, _T_TYPE_& dst)
{
	_Tensor<T>& _src = dynamic_cast<_Tensor<T>&>(src);
	_Tensor<T>& _dst = dynamic_cast<_Tensor<T>&>(dst);

	de::DH handle;
	if (!cuP.is_init) {
		Not_init(&handle);
		return handle;
	}

	if (_src.width != _dst.width || _src.height != _dst.height) {
		MDim_Not_Matching(&handle);
		return handle;
	}

	uint alloc_d_width = _A.width * _A.pitch;
	Kadd_c(_src.Mat, __x, _dst.Mat, alloc_d_width, _src.height, alloc_d_width, alloc_d_width);

	Success(&handle);
	return handle;
}

template _DECX_API_ de::DH de::Add(T_FLOAT& src, float __x, T_FLOAT& dst);

template _DECX_API_ de::DH de::Add(T_INT& src, int __x, T_INT& dst);

template _DECX_API_ de::DH de::Add(T_HALF& src, de::Half __x, T_HALF& dst);

template _DECX_API_ de::DH de::Add(T_DOUBLE& src, double __x, T_DOUBLE& dst);