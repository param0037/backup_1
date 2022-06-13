/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _CONV2_H_
#define _CONV2_H_

#include "../../../classes/MatrixArray.h"
#include "../../../classes/GPU_Matrix.h"
#include "conv_utils.h"
#include "fp32/conv2_border_ignored_fp32.h"
#include "fp16/conv2_border_ignore_fp16.h"
#include "fp32/conv2_border_const_fp32.h"
#include "fp16/conv2_border_const_fp16.h"
#include "fp32/dev_conv2_border_const_fp32.h"
#include "fp32/dev_conv2_border_ignored_fp32.h"
#include "fp16/dev_conv2_border_const_fp16.h"
#include "fp16/dev_conv2_border_ignored_fp16.h"


namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH Conv2(de::Matrix<float>& src, de::Matrix<float>& kernel, de::Matrix<float>& dst, const uint flag);



        _DECX_API_ de::DH Conv2(de::Matrix<de::Half>& src, de::Matrix<de::Half>& kernel, de::Matrix<de::Half>& dst, const uint flag);



        _DECX_API_ de::DH Conv2(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& kernel, de::GPU_Matrix<float>& dst, const uint flag);



        _DECX_API_ de::DH Conv2(de::GPU_Matrix<de::Half>& src, de::GPU_Matrix<de::Half>& kernel, de::GPU_Matrix<de::Half>& dst, const uint flag);
    }
}




de::DH de::cuda::Conv2(_FLOAT_& src, _FLOAT_& kernel, _FLOAT_& dst, const uint flag)
{
    decx::_Matrix<float>& _src = dynamic_cast<decx::_Matrix<float>&>(src);
    decx::_Matrix<float>& _kernel = dynamic_cast<decx::_Matrix<float>&>(kernel);
    decx::_Matrix<float>& _dst = dynamic_cast<decx::_Matrix<float>&>(dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::sConv2_border_ignore(_src, _kernel, _dst, &handle);
        break;
    case decx::conv_property::de_conv_zero_compensate:
        decx::sConv2_border_zero(_src, _kernel, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}



de::DH de::cuda::Conv2(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& kernel, de::GPU_Matrix<float>& dst, const uint flag)
{
    decx::_GPU_Matrix<float>& _src = dynamic_cast<decx::_GPU_Matrix<float>&>(src);
    decx::_GPU_Matrix<float>& _kernel = dynamic_cast<decx::_GPU_Matrix<float>&>(kernel);
    decx::_GPU_Matrix<float>& _dst = dynamic_cast<decx::_GPU_Matrix<float>&>(dst);

    de::DH handle;

    decx::Success(&handle);

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::dev_sConv2_border_ignore(_src, _kernel, _dst, &handle);
        break;
    case decx::conv_property::de_conv_zero_compensate:
        decx::dev_sConv2_border_zero(_src, _kernel, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}



de::DH de::cuda::Conv2(de::GPU_Matrix<de::Half>& src, de::GPU_Matrix<de::Half>& kernel, de::GPU_Matrix<de::Half>& dst, const uint flag)
{
    decx::_GPU_Matrix<de::Half>& _src = dynamic_cast<decx::_GPU_Matrix<de::Half>&>(src);
    decx::_GPU_Matrix<de::Half>& _kernel = dynamic_cast<decx::_GPU_Matrix<de::Half>&>(kernel);
    decx::_GPU_Matrix<de::Half>& _dst = dynamic_cast<decx::_GPU_Matrix<de::Half>&>(dst);

    de::DH handle;

    decx::Success(&handle);

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::dev_hConv2_border_ignore(_src, _kernel, _dst, &handle);
        break;
    case decx::conv_property::de_conv_zero_compensate:
        decx::dev_hConv2_border_zero(_src, _kernel, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}



de::DH de::cuda::Conv2(_HALF_& src, _HALF_& kernel, _HALF_& dst, const uint flag)
{
    decx::_Matrix<de::Half>& _src = dynamic_cast<decx::_Matrix<de::Half>&>(src);
    decx::_Matrix<de::Half>& _kernel = dynamic_cast<decx::_Matrix<de::Half>&>(kernel);
    decx::_Matrix<de::Half>& _dst = dynamic_cast<decx::_Matrix<de::Half>&>(dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::hConv2_border_ignore(_src, _kernel, _dst, &handle);
        break;
    case decx::conv_property::de_conv_zero_compensate:
        decx::hConv2_border_zero(_src, _kernel, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}


#endif        // #ifndef _CONV2_H_