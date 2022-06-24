/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _CV_CLS_MFUNCS_H_
#define _CV_CLS_MFUNCS_H_

#include "../../core/basic.h"
#include "../../classes/matrix.h"
#include "../cv_classes/cv_classes.h"
#include "../../core/memory_management/MemBlock.h"
#include "../../handles/decx_handles.h"
#ifdef _DECX_CPU_CODES_
#include "../utils/cvt_colors.h"
#endif




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

#ifdef _DECX_CPU_CODES_
        _DECX_API_ de::DH merge_channel(de::vis::Img& src, de::vis::Img& dst, const int flag);
#endif
    }
}

#ifdef _DECX_CPU_CODES_
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

    case de::vis::ImgChannelMergeType::Preserve_Alpha:
        decx::_Preserve_A_UC2UC_caller(
            reinterpret_cast<float*>(_src->Mat.ptr), reinterpret_cast<float*>(_dst->Mat.ptr), make_int2(_src->pitch, _src->height));
        break;

    default:
        break;
    }
    

    return handle;
}
#endif

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