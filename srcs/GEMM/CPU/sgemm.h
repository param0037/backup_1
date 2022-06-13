/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _SGEMM_H_
#define _SGEMM_H_

#include "sgemm_callers.h"
#include "../../classes/Matrix.h"


namespace de
{
    namespace cpu
    {
        _DECX_API_ de::DH sgemm(de::Matrix<float>& A, de::Matrix<float>& B, de::Matrix<float>& dst);
    }
}



de::DH de::cpu::sgemm(de::Matrix<float>& A, de::Matrix<float>& B, de::Matrix<float>& dst)
{
    decx::_Matrix<float>* _A = dynamic_cast<decx::_Matrix<float>*>(&A);
    decx::_Matrix<float>* _B = dynamic_cast<decx::_Matrix<float>*>(&B);
    decx::_Matrix<float>* _dst = dynamic_cast<decx::_Matrix<float>*>(&dst);

    de::DH handle;
    if (_A->width != _B->height) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    _dst->re_construct(_A->width, _B->height, decx::DATA_STORE_TYPE::Page_Default);
    
    int4 dims_info = make_int4(_A->pitch, _A->height, _dst->pitch, 0);
    int2 sort_dims_info = make_int2(_A->pitch, _B->pitch);
    
    decx::PtrInfo<float> B_buffer;
    int res = decx::alloc::_host_virtual_page_malloc<float>(&B_buffer,
        static_cast<size_t>(_B->pitch) * static_cast<size_t>(_A->pitch) * sizeof(float));
    if (res) {
        decx::err::AllocateFailure(&handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return handle;
    }

    decx::sgemm_caller<3, 4>(_A->Mat.ptr, _B->Mat.ptr, B_buffer.ptr, _dst->Mat.ptr, &dims_info);
    
    decx::alloc::_dealloc_Hv(B_buffer.block);

    decx::Success(&handle);
    return handle;
}

#endif