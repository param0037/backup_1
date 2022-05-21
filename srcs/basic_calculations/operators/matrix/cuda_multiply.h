/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.9.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.9.16
*/

/**
* 处理思想： 只拷贝和处理有效的行数，但是列数必须全部处理
*        x256
* ---------------
* |*********....|
* |*********....|
* |*********....|    x128
* |*********....|
* |                |
* ---------------
* where "*" represents effective elements and "." represents idle elements
*/

#pragma once

#include "../../../classes/matrix.h"
#include "../../../classes/GPU_Matrix.h"
#include "../Mul_kernel.cuh"
#include "../../../core/basic.h"

using decx::_Matrix;
using decx::_GPU_Matrix;

namespace de
{
    namespace cuda
    {
        template <typename T>
        _DECX_API_  de::DH Mul(de::GPU_Matrix<T>& A, de::GPU_Matrix<T>& B, de::GPU_Matrix<T>& dst);



        template <typename T>
        _DECX_API_  de::DH Mul(de::GPU_Matrix<T>& src, T __x, de::GPU_Matrix<T>& dst);
    }
}



template <typename T>
de::DH de::cuda::Mul(de::GPU_Matrix<T>& A, de::GPU_Matrix<T>& B, de::GPU_Matrix<T>& dst)
{
    _GPU_Matrix<T>& _A = dynamic_cast<_GPU_Matrix<T>&>(A);
    _GPU_Matrix<T>& _B = dynamic_cast<_GPU_Matrix<T>&>(B);
    _GPU_Matrix<T>& _dst = dynamic_cast<_GPU_Matrix<T>&>(dst);

    de::DH handle;
    if (!cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_A.width != _B.width || _A.height != _B.height) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    const size_t len = (size_t)_A._length;
    decx::dev_Kmul_m(_A.Mat, _B.Mat, _dst.Mat, len);

    decx::Success(&handle);
    return handle;
}

template _DECX_API_ de::DH de::cuda::Mul(de::GPU_Matrix<float>& A, de::GPU_Matrix<float>& B, de::GPU_Matrix<float>& dst);

template _DECX_API_ de::DH de::cuda::Mul(de::GPU_Matrix<int>& A, de::GPU_Matrix<int>& B, de::GPU_Matrix<int>& dst);

template _DECX_API_ de::DH de::cuda::Mul(de::GPU_Matrix<de::Half>& A, de::GPU_Matrix<de::Half>& B, de::GPU_Matrix<de::Half>& dst);

template _DECX_API_ de::DH de::cuda::Mul(de::GPU_Matrix<double>& A, de::GPU_Matrix<double>& B, de::GPU_Matrix<double>& dst);





template <typename T>
de::DH de::cuda::Mul(de::GPU_Matrix<T>& src, T __x, de::GPU_Matrix<T>& dst)
{
    _GPU_Matrix<T>& _src = dynamic_cast<_GPU_Matrix<T>&>(src);
    _GPU_Matrix<T>& _dst = dynamic_cast<_GPU_Matrix<T>&>(dst);

    de::DH handle;
    if (!cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_src.width != _dst.width || _src.height != _dst.height) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    const size_t len = (size_t)_src._length;
    decx::dev_Kmul_c(_src.Mat, __x, _dst.Mat, len);

    decx::Success(&handle);
    return handle;
}

template _DECX_API_ de::DH de::cuda::Mul(de::GPU_Matrix<float>& src, float __x, de::GPU_Matrix<float>& dst);

template _DECX_API_ de::DH de::cuda::Mul(de::GPU_Matrix<int>& src, int __x, de::GPU_Matrix<int>& dst);

template _DECX_API_ de::DH de::cuda::Mul(de::GPU_Matrix<de::Half>& src, de::Half __x, de::GPU_Matrix<de::Half>& dst);

template _DECX_API_ de::DH de::cuda::Mul(de::GPU_Matrix<double>& src, double __x, de::GPU_Matrix<double>& dst);