#pragma once
#include "../matrix/add.h"
#include "../../../classes/Matrix_member_function_definition.h"


namespace de
{
	template <typename T>
	de::DH Add(DT_TYPE_& A, DT_TYPE_& B, DT_TYPE_& dst);


	template <typename T>
	de::DH Add(DT_TYPE_& src, T __x, DT_TYPE_& dst);



	de::DH Add(DT_HALF& A, DT_HALF& B, DT_HALF& dst);


	
	de::DH Add(DT_HALF& src, de::Half __x, DT_HALF& dst);
}



template <typename T>
de::DH de::Add(DT_TYPE_& A, DT_TYPE_& B, DT_TYPE_& dst)
{
	de::DH handle;
	if (!cuP.is_init) {
		Not_init(&handle);
		return handle;
	}

	__dev_Tensor<T>* _A = dynamic_cast<__dev_Tensor<T>*>(&A);
	__dev_Tensor<T>* _B = dynamic_cast<__dev_Tensor<T>*>(&B);
	__dev_Tensor<T>* _dst = dynamic_cast<__dev_Tensor<T>*>(&dst);

	const uint A_width = _A->width;
	const uint A_height = _A->height;
	const uint A_depth = _A->depth;

	const uint B_width = _B->width;
	const uint B_height = _B->height;
	const uint B_depth = _B->depth;

	if (A_width != B_width || A_height != B_height || A_depth != B_depth) {
		MDim_Not_Matching(&handle);
		return handle;
	}
	const size_t total = _A->element_num;
	dev_Kadd_MM<T>(_A->dev_Tens, _B->dev_Tens, _dst->dev_Tens, &total);

	Success(&handle);
	return handle;
}

template _DECX_API_ de::DH de::Add(DT_INT &A, DT_INT &B, DT_INT &dst);

template _DECX_API_ de::DH de::Add(DT_FLOAT& A, DT_FLOAT& B, DT_FLOAT& dst);

template _DECX_API_ de::DH de::Add(DT_DOUBLE& A, DT_DOUBLE& B, DT_DOUBLE& dst);



template <typename T>
de::DH de::Add(DT_TYPE_& src, T __x, DT_TYPE_& dst)
{
	de::DH handle;
	if (!cuP.is_init) {
		Not_init(&handle);
		return handle;
	}

	__dev_Tensor<T>* _src = dynamic_cast<__dev_Tensor<T>*>(&src);
	__dev_Tensor<T>* _dst = dynamic_cast<__dev_Tensor<T>*>(&dst);

	const size_t total = _src->element_num;
	dev_Kadd_C<T>(_src->dev_Tens, &__x, _dst->dev_Tens, &total);

	Success(&handle);
	return handle;
}

template _DECX_API_ de::DH de::Add(DT_INT& src, int __X, DT_INT& dst);

template _DECX_API_ de::DH de::Add(DT_FLOAT& src, float __x, DT_FLOAT& dst);

template _DECX_API_ de::DH de::Add(DT_DOUBLE& src, double __x, DT_DOUBLE& dst);


// ---------------------- half ---------------------------------------


de::DH de::Add(DT_HALF& A, DT_HALF& B, DT_HALF& dst)
{
	de::DH handle;
	if (!cuP.is_init) {
		Not_init(&handle);
		return handle;
	}

	__dev_Tensor<de::Half>* _A = dynamic_cast<__dev_Tensor<de::Half>*>(&A);
	__dev_Tensor<de::Half>* _B = dynamic_cast<__dev_Tensor<de::Half>*>(&B);
	__dev_Tensor<de::Half>* _dst = dynamic_cast<__dev_Tensor<de::Half>*>(&dst);

	const uint A_width = _A->width;
	const uint A_height = _A->height;
	const uint A_depth = _A->depth;

	const uint B_width = _B->width;
	const uint B_height = _B->height;
	const uint B_depth = _B->depth;

	if (A_width != B_width || A_height != B_height || A_depth != B_depth) {
		MDim_Not_Matching(&handle);
		return handle;
	}
	const size_t total = _A->element_num;
	dev_Kadd_MMH(reinterpret_cast<int*>(_A->dev_Tens), 
		reinterpret_cast<int*>(_B->dev_Tens), reinterpret_cast<int*>(_dst->dev_Tens), &total);

	Success(&handle);
	return handle;
}



de::DH Add(DT_HALF& src, de::Half __x, DT_HALF& dst)
{
	de::DH handle;
	if (!cuP.is_init) {
		Not_init(&handle);
		return handle;
	}

	__dev_Tensor<de::Half>* _src = dynamic_cast<__dev_Tensor<de::Half>*>(&src);
	__dev_Tensor<de::Half>* _dst = dynamic_cast<__dev_Tensor<de::Half>*>(&dst);

	const size_t total = _src->element_num;
	dev_Kadd_CH(reinterpret_cast<int*>(_src->dev_Tens), &__x, reinterpret_cast<int*>(_dst->dev_Tens), &total);

	Success(&handle);
	return handle;
}