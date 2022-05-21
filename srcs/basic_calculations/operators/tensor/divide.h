// ==================== Tensor ===========================

#pragma once
#include "../matrix/divide.h"


namespace de
{
	template <typename T>
	de::DH Div(_T_TYPE_& A, _T_TYPE_& B, _T_TYPE_& dst);


	template <typename T>
	de::DH Div(_T_TYPE_& src, T __x, _T_TYPE_& dst);


	template <typename T>
	de::DH Div(T __x, _T_TYPE_& src, _T_TYPE_& dst);


	_DECX_API_
	de::DH Div_fp16(T_HALF& A, T_HALF& B, T_HALF& dst);



	_DECX_API_
	de::DH Div_fp16(T_HALF& src, de::Half __x, T_HALF& dst);



	_DECX_API_
	de::DH Div_fp16(de::Half __x, T_HALF& src, T_HALF& dst);
}



template <typename T>
de::DH de::Div(_T_TYPE_& A, _T_TYPE_& B, _T_TYPE_& dst)
{
	de::DH handle;
	if (!(cuP.is_init)) {
		Not_init(&handle);
		return handle;
	}

	_Tensor<T>& _A = dynamic_cast<_Tensor<T>&>(A);
	_Tensor<T>& _B = dynamic_cast<_Tensor<T>&>(B);
	_Tensor<T>& _dst = dynamic_cast<_Tensor<T>&>(dst);

	const uint A_width = _A.width;
	const uint A_height = _A.height;
	const uint A_depth = _A.depth;

	const uint B_width = _B.width;
	const uint B_height = _B.height;
	const uint B_depth = _B.depth;

	if (A_width != B_width || A_height != B_height || A_depth != B_depth) {
		MDim_Not_Matching(&handle);
		return handle;
	}
	size_t total = _A.element_num;
	Kdiv_MM<T>(_A.Tens, _B.Tens, _dst.Tens, &total);

	Success(&handle);
	return handle;
}

template de::DH _DECX_API_ de::Div(T_INT& A, T_INT& B, T_INT& dst);

template de::DH _DECX_API_ de::Div(T_FLOAT& A, T_FLOAT& B, T_FLOAT& dst);

template de::DH _DECX_API_ de::Div(T_DOUBLE& A, T_DOUBLE& B, T_DOUBLE& dst);




template <typename T>
de::DH de::Div(_T_TYPE_& src, T __x, _T_TYPE_& dst)
{
	de::DH handle;
	if (!(cuP.is_init)) {
		Not_init(&handle);
		return handle;
	}

	_Tensor<T>& _src = dynamic_cast<_Tensor<T>&>(src);
	_Tensor<T>& _dst = dynamic_cast<_Tensor<T>&>(dst);

	size_t total = _src.element_num;
	Kdiv_MMC<T>(_src.Tens, &__x, _dst.Tens, &total);

	Success(&handle);
	return handle;
}

template de::DH _DECX_API_ de::Div(T_INT& src, int __x, T_INT& dst);

template de::DH _DECX_API_ de::Div(T_FLOAT& src, float __x, T_FLOAT& dst);

template de::DH _DECX_API_ de::Div(T_DOUBLE& src, double __x, T_DOUBLE& dst);



template <typename T>
de::DH de::Div(T __x, _T_TYPE_& src, _T_TYPE_& dst)
{
	de::DH handle;
	if (!(cuP.is_init)) {
		Not_init(&handle);
		return handle;
	}

	_Tensor<T>& _src = dynamic_cast<_Tensor<T>&>(src);
	_Tensor<T>& _dst = dynamic_cast<_Tensor<T>&>(dst);

	size_t total = _src.element_num;
	Kdiv_MMCinv<T>(_src.Tens, &__x, _dst.Tens, &total);

	Success(&handle);
	return handle;
}

template de::DH _DECX_API_ de::Div(int __x, T_INT& src, T_INT& dst);

template de::DH _DECX_API_ de::Div(float __x, T_FLOAT& src, T_FLOAT& dst);

template de::DH _DECX_API_ de::Div(double __x, T_DOUBLE& src, T_DOUBLE& dst);



de::DH de::Div_fp16(T_HALF& A, T_HALF& B, T_HALF& dst)
{
	de::DH handle;
	if (!(cuP.is_init)) {
		Not_init(&handle);
		return handle;
	}

	_Tensor<de::Half>& _A = dynamic_cast<_Tensor<de::Half>&>(A);
	_Tensor<de::Half>& _B = dynamic_cast<_Tensor<de::Half>&>(B);
	_Tensor<de::Half>& _dst = dynamic_cast<_Tensor<de::Half>&>(dst);

	const uint A_width = _A.width;
	const uint A_height = _A.height;
	const uint A_depth = _A.depth;

	const uint B_width = _B.width;
	const uint B_height = _B.height;
	const uint B_depth = _A.depth;

	if (A_width != B_width || A_height != B_height || A_depth != B_depth) {
		MDim_Not_Matching(&handle);
		return handle;
	}
	const size_t total = _A.element_num;
	Kdiv_MMH(_A.Tens, _B.Tens, _dst.Tens, &total);

	Success(&handle);
	return handle;
}



de::DH de::Div_fp16(T_HALF& src, de::Half __x, T_HALF& dst)
{
	de::DH handle;
	if (!(cuP.is_init)) {
		Not_init(&handle);
		return handle;
	}

	_Tensor<de::Half>& _src = dynamic_cast<_Tensor<de::Half>&>(src);
	_Tensor<de::Half>& _dst = dynamic_cast<_Tensor<de::Half>&>(dst);

	const size_t total = _src.element_num;
	Kdiv_MMCH(_src.Tens, &__x, _dst.Tens, &total);

	Success(&handle);
	return handle;
}




de::DH de::Div_fp16(de::Half __x, T_HALF& src, T_HALF& dst)
{
	de::DH handle;
	if (!(cuP.is_init)) {
		Not_init(&handle);
		return handle;
	}

	_Tensor<de::Half>& _src = dynamic_cast<_Tensor<de::Half>&>(src);
	_Tensor<de::Half>& _dst = dynamic_cast<_Tensor<de::Half>&>(dst);

	const size_t total = _src.element_num;
	Kdiv_MMCinvH(_src.Tens, &__x, _dst.Tens, &total);

	Success(&handle);
	return handle;
}