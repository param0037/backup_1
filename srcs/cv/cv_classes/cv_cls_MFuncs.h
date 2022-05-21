/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#ifndef _CV_CLS_MFUNCS_H_
#define _CV_CLS_MFUNCS_H_

#include "../../core/basic.h"
#include "../../classes/matrix.h"
#include "cv_classes.h"
#include "../../core/memory_management/MemBlock.h"
#include "../../handles/decx_handles.h"
#include "../utils/cvt_colors_def.h"


/*
* I choose __m128 to load the uchar image array instead of __m256. If I use __m256 as the stride,
* register will be overloaded
*/
#define _IMG_ALIGN_ 16

decx::_Img::_Img(const uint width, const uint height, const int flag)
{
    this->height = height;
    this->width = width;
    this->ImgPlane = static_cast<size_t>(this->width) * static_cast<size_t>(this->height);

    switch (flag)
    {
    case de::vis::ImgConstructType::DE_UC1:        // UC1
        this->channel = 1;
        this->pitch = decx::utils::ceil<uint>(width, _IMG_ALIGN_) * _IMG_ALIGN_;
        break;

    case de::vis::ImgConstructType::DE_UC3:        // UC3
        this->channel = 4;
        this->pitch = decx::utils::ceil<uint>(width, _IMG_ALIGN_) * _IMG_ALIGN_;
        break;

    case de::vis::ImgConstructType::DE_UC4:        // UC4
        this->channel = 4;
        this->pitch = decx::utils::ceil<uint>(width, _IMG_ALIGN_) * _IMG_ALIGN_;
        break;

    default:    // default situation : align up with 4 bytes(uchar4)
        this->pitch = decx::utils::ceil<uint>(width, _IMG_ALIGN_) * _IMG_ALIGN_;
        this->channel = 4;
        break;
    }
    
    this->element_num = static_cast<size_t>(this->channel) * this->ImgPlane;
    this->total_bytes = this->element_num * sizeof(uchar);
    this->_element_num = (size_t)this->pitch * (size_t)this->height * (size_t)this->channel;
    this->_total_bytes = this->_element_num * sizeof(uchar);

#ifndef GNU_CPUcodes
    if (decx::alloc::_host_virtual_page_malloc(&this->Mat, this->_total_bytes)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
#else
    if (decx::alloc::_host_virtual_page_malloc(&this->Mat, this->total_bytes)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
#endif
}



uchar* decx::_Img::Ptr(const uint row, const uint col)
{
    return (this->Mat.ptr + ((size_t)row * (size_t)(this->pitch) + col) * (size_t)(this->channel));
}





void decx::_Img::release()
{
#ifndef GNU_CPUcodes
    decx::alloc::_host_virtual_page_dealloc(&this->Mat);
#else
    decx::alloc::_host_virtual_page_dealloc(&this->Mat);
#endif
}



namespace de
{
    namespace vis
    {
        enum ImgChannelMergeType
        {
            BGR_to_Gray = 0,
            Preserve_B = 1,
            Preserve_G = 2,
            Preserve_R = 3,
            Preserve_Alpha = 4,
            RGB_mean = 5,
        };

        _DECX_API_ de::DH merge_channel(de::vis::Img& src, de::vis::Img& dst, const int flag);
    }
}


de::DH de::vis::merge_channel(de::vis::Img& src, de::vis::Img& dst, const int flag)
{
    de::DH handle;
    decx::_Img* _src = dynamic_cast<decx::_Img*>(&src);
    decx::_Img* _dst = dynamic_cast<decx::_Img*>(&dst);

    switch (flag)
    {
    case de::vis::ImgChannelMergeType::BGR_to_Gray:
        decx::_BGR2Gray_UC2UC_caller(
            reinterpret_cast<float*>(_src->Mat.ptr), reinterpret_cast<float*>(_dst->Mat.ptr), make_int2(_src->pitch, _src->height));
        break;

    case de::vis::ImgChannelMergeType::Preserve_B:
        decx::_Preserve_B_UC2UC_caller(
            reinterpret_cast<float*>(_src->Mat.ptr), reinterpret_cast<float*>(_dst->Mat.ptr), make_int2(_src->pitch, _src->height));
        break;

    case de::vis::ImgChannelMergeType::Preserve_G:
        decx::_Preserve_G_UC2UC_caller(
            reinterpret_cast<float*>(_src->Mat.ptr), reinterpret_cast<float*>(_dst->Mat.ptr), make_int2(_src->pitch, _src->height));
        break;

    case de::vis::ImgChannelMergeType::Preserve_R:
        decx::_Preserve_R_UC2UC_caller(
            reinterpret_cast<float*>(_src->Mat.ptr), reinterpret_cast<float*>(_dst->Mat.ptr), make_int2(_src->pitch, _src->height));
        break;

    default:
        break;
    }
    

    return handle;
}


namespace de
{
    namespace vis
    {
        _DECX_API_ de::vis::Img *CreateImgPtr(const uint width, const uint height, const int flag);


        _DECX_API_ de::vis::Img* CreateImgPtr();


        _DECX_API_ de::vis::Img &CreateImgRef(const uint width, const uint height, const int flag);


        _DECX_API_ de::vis::Img& CreateImgRef();
    }
}



de::vis::Img* de::vis::CreateImgPtr(const uint width, const uint height, const int flag)
{
    return new decx::_Img(width, height, flag);
}



de::vis::Img* de::vis::CreateImgPtr()
{
    return new decx::_Img();
}



de::vis::Img& de::vis::CreateImgRef(const uint width, const uint height, const int flag)
{
    return *(new decx::_Img(width, height, flag));
}



de::vis::Img& de::vis::CreateImgRef()
{
    return *(new decx::_Img());
}

#endif