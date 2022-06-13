/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#include "../../../classes/vector.h"
#include "../../../classes/GPU_Vector.h"
#include "../Mul_kernel.cuh"
#include "../../../core/basic.h"

using decx::_Vector;
using decx::_GPU_Vector;


namespace de
{
    namespace cuda
    {
        template <typename T>
        _DECX_API_  de::DH Mul(de::GPU_Vector<T>& A, de::GPU_Vector<T>& B, de::GPU_Vector<T>& dst);



        template <typename T>
        _DECX_API_  de::DH Mul(de::GPU_Vector<T>& src, T __x, de::GPU_Vector<T>& dst);
    }
}


using decx::_GPU_Vector;


template <typename T>
de::DH de::cuda::Mul(de::GPU_Vector<T>& A, de::GPU_Vector<T>& B, de::GPU_Vector<T>& dst)
{
    _GPU_Vector<T>& _A = dynamic_cast<_GPU_Vector<T>&>(A);
    _GPU_Vector<T>& _B = dynamic_cast<_GPU_Vector<T>&>(B);
    _GPU_Vector<T>& _dst = dynamic_cast<_GPU_Vector<T>&>(dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_A._length != _B._length) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    const size_t len = (size_t)_A._length;
    decx::dev_Kmul_m(_A.Vec.ptr, _B.Vec.ptr, _dst.Vec.ptr, len);

    decx::Success(&handle);
    return handle;
}

template _DECX_API_ de::DH de::cuda::Mul(de::GPU_Vector<float>& A, de::GPU_Vector<float>& B, de::GPU_Vector<float>& dst);

template _DECX_API_ de::DH de::cuda::Mul(de::GPU_Vector<int>& A, de::GPU_Vector<int>& B, de::GPU_Vector<int>& dst);

template _DECX_API_ de::DH de::cuda::Mul(de::GPU_Vector<de::Half>& A, de::GPU_Vector<de::Half>& B, de::GPU_Vector<de::Half>& dst);

template _DECX_API_ de::DH de::cuda::Mul(de::GPU_Vector<double>& A, de::GPU_Vector<double>& B, de::GPU_Vector<double>& dst);





template <typename T>
de::DH de::cuda::Mul(de::GPU_Vector<T>& A, T __x, de::GPU_Vector<T>& dst)
{
    _GPU_Vector<T>& _A = dynamic_cast<_GPU_Vector<T>&>(A);
    _GPU_Vector<T>& _dst = dynamic_cast<_GPU_Vector<T>&>(dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_A.length != _dst.length) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    const size_t len = (size_t)_A._length;
    decx::dev_Kmul_c(_A.Vec.ptr, __x,  _dst.Vec.ptr, len);

    decx::Success(&handle);
    return handle;
}

template _DECX_API_ de::DH de::cuda::Mul(de::GPU_Vector<float>& A, float __x, de::GPU_Vector<float>& dst);

template _DECX_API_ de::DH de::cuda::Mul(de::GPU_Vector<int>& A, int __x, de::GPU_Vector<int>& dst);

template _DECX_API_ de::DH de::cuda::Mul(de::GPU_Vector<de::Half>& A, de::Half __x, de::GPU_Vector<de::Half>& dst);

template _DECX_API_ de::DH de::cuda::Mul(de::GPU_Vector<double>& A, double __x, de::GPU_Vector<double>& dst);