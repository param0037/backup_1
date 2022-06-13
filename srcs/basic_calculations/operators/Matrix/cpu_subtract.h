/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _CPU_SUB_MATRIX_H_
#define _CPU_SUB_MATRIX_H_

#include "../../../classes/Matrix.h"
#include "../Sub_exec.h"

using decx::_Matrix;

namespace de
{
    namespace cpu
    {
        template <typename T>
        _DECX_API_ de::DH Sub(de::Matrix<T>& A, de::Matrix<T>& B, de::Matrix<T>& dst);


        template <typename T>
        _DECX_API_ de::DH Sub(de::Matrix<T>& src, const T B, de::Matrix<T>& dst);


        template <typename T>
        _DECX_API_ de::DH Sub(const T B, de::Matrix<T>& src, de::Matrix<T>& dst);
    }
}


template <typename T>
de::DH de::cpu::Sub(de::Matrix<T>& A, de::Matrix<T>& B, de::Matrix<T>& dst)
{
    _Matrix<T>* _A = dynamic_cast<_Matrix<T>*>(&A);
    _Matrix<T>* _B = dynamic_cast<_Matrix<T>*>(&B);
    _Matrix<T>* _dst = dynamic_cast<_Matrix<T>*>(&dst);

    de::DH handle;
    decx::Success(&handle);

    if (!decx::cpI.is_init) {
        decx::Not_init(&handle);
        Print_Error_Message(4, NOT_INIT);
        exit(-1);
    }
    
    decx::Ksub_m(_A->Mat.ptr, _B->Mat.ptr, _dst->Mat.ptr, _A->_element_num);

    return handle;
}

template _DECX_API_ de::DH de::cpu::Sub(de::Matrix<float>& A, de::Matrix<float>& B, de::Matrix<float>& dst);

template _DECX_API_ de::DH de::cpu::Sub(de::Matrix<int>& A, de::Matrix<int>& B, de::Matrix<int>& dst);

template _DECX_API_ de::DH de::cpu::Sub(de::Matrix<double>& A, de::Matrix<double>& B, de::Matrix<double>& dst);



template <typename T>
de::DH de::cpu::Sub(de::Matrix<T>& src, const T __x, de::Matrix<T>& dst)
{
    _Matrix<T>* _src = dynamic_cast<_Matrix<T>*>(&src);
    _Matrix<T>* _dst = dynamic_cast<_Matrix<T>*>(&dst);

    de::DH handle;
    decx::Success(&handle);

    if (!decx::cpI.is_init) {
        decx::Not_init(&handle);
        Print_Error_Message(4, NOT_INIT);
        exit(-1);
    }

    decx::Ksub_c(_src->Mat.ptr, __x, _dst->Mat.ptr, _src->_element_num);

    return handle;
}

template _DECX_API_ de::DH de::cpu::Sub(de::Matrix<float>& src, const float __x, de::Matrix<float>& dst);

template _DECX_API_ de::DH de::cpu::Sub(de::Matrix<int>& src, const int __x, de::Matrix<int>& dst);

template _DECX_API_ de::DH de::cpu::Sub(de::Matrix<double>& src, const double __x, de::Matrix<double>& dst);



template <typename T>
de::DH de::cpu::Sub(const T __x, de::Matrix<T>& src, de::Matrix<T>& dst)
{
    _Matrix<T>* _src = dynamic_cast<_Matrix<T>*>(&src);
    _Matrix<T>* _dst = dynamic_cast<_Matrix<T>*>(&dst);

    de::DH handle;
    decx::Success(&handle);

    if (!decx::cpI.is_init) {
        decx::Not_init(&handle);
        Print_Error_Message(4, NOT_INIT);
        exit(-1);
    }

    decx::Ksub_cinv(_src->Mat.ptr, __x, _dst->Mat.ptr, _src->_element_num);

    return handle;
}

template _DECX_API_ de::DH de::cpu::Sub(const float __x, de::Matrix<float>& src, de::Matrix<float>& dst);

template _DECX_API_ de::DH de::cpu::Sub(const int __x, de::Matrix<int>& src, de::Matrix<int>& dst);

template _DECX_API_ de::DH de::cpu::Sub(const double __x, de::Matrix<double>& src, de::Matrix<double>& dst);

#endif