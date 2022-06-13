/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _CONV2_MC_H_
#define _CONV2_MC_H_

// classes
#include "../../../classes/MatrixArray.h"
#include "../../../core/basic.h"
#include "../../../classes/GPU_Matrix.h"
#include "../../../classes/GPU_MatrixArray.h"

// sub-functions
#include "fp32/conv2_border_ignored_SK_fp32.h"
#include "fp16/conv2_border_ignored_fp16_SK.h"
#include "fp32/conv2_border_const_SK_fp32.h"
#include "fp16/conv2_border_const_SK_fp16.h"
#include "fp32/conv2_border_ignored_MK_fp32.h"
#include "fp16/conv2_border_ignored_fp16_MK.h"
#include "fp32/conv2_border_const_MK_fp32.h"
#include "fp16/conv2_border_const_MK_fp16.h"

#include "fp32/dev_conv2_border_ignored_SK_fp32.h"
#include "fp32/dev_conv2_border_const_SK_fp32.h"
#include "fp32/dev_conv2_border_const_MK_fp32.h"
#include "fp32/dev_conv2_border_ignored_MK_fp32.h"

#include "fp16/dev_conv2_border_ignored_SK_fp16.h"
#include "fp16/dev_conv2_border_const_SK_fp16.h"
#include "fp16/dev_conv2_border_const_MK_fp16.h"
#include "fp16/dev_conv2_border_ignored_MK_fp16.h"

#include "Conv2.h"


namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH Conv2_single_kernel(de::MatrixArray<float>& src, de::Matrix<float>& kernel, de::MatrixArray<float>& dst, const int flag);



        _DECX_API_ de::DH Conv2_single_kernel(de::MatrixArray<de::Half>& src, de::Matrix<de::Half>& kernel, de::MatrixArray<de::Half>& dst, const int flag);



        _DECX_API_ de::DH Conv2_multi_kernel(de::MatrixArray<float>& src, de::MatrixArray<float>& kernel, de::MatrixArray<float>& dst, const int flag);



        _DECX_API_ de::DH Conv2_multi_kernel(de::MatrixArray<de::Half>& src, de::MatrixArray<de::Half>& kernel, de::MatrixArray<de::Half>& dst, const int flag);


        // on device
        _DECX_API_ de::DH Conv2_single_kernel(de::GPU_MatrixArray<float>& src, de::GPU_Matrix<float>& kernel, de::GPU_MatrixArray<float>& dst, const int flag);



        _DECX_API_ de::DH Conv2_multi_kernel(de::GPU_MatrixArray<float>& src, de::GPU_MatrixArray<float>& kernel, de::GPU_MatrixArray<float>& dst, const int flag);



        _DECX_API_ de::DH Conv2_single_kernel(de::GPU_MatrixArray<de::Half>& src, de::GPU_Matrix<de::Half>& kernel, de::GPU_MatrixArray<de::Half>& dst, const int flag);



        _DECX_API_ de::DH Conv2_multi_kernel(de::GPU_MatrixArray<de::Half>& src, de::GPU_MatrixArray<de::Half>& kernel, de::GPU_MatrixArray<de::Half>& dst, const int flag);
    }
}


using decx::_MatrixArray;


de::DH de::cuda::Conv2_single_kernel(de::MatrixArray<float>& src, de::Matrix<float>& kernel, de::MatrixArray<float>& dst, const int flag)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    _MatrixArray<float>* _src = dynamic_cast<_MatrixArray<float>*>(&src);
    decx::_Matrix<float>* _kernel = dynamic_cast<decx::_Matrix<float>*>(&kernel);
    _MatrixArray<float>* _dst = dynamic_cast<_MatrixArray<float>*>(&dst);

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::sConv2_border_ignore_sk(_src, _kernel, _dst, &handle);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        decx::sConv2_border_zero_sk(_src, _kernel, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}





de::DH de::cuda::Conv2_single_kernel(de::MatrixArray<de::Half>& src, de::Matrix<de::Half>& kernel, de::MatrixArray<de::Half>& dst, const int flag)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    _MatrixArray<de::Half>* _src = dynamic_cast<_MatrixArray<de::Half>*>(&src);
    decx::_Matrix<de::Half>* _kernel = dynamic_cast<decx::_Matrix<de::Half>*>(&kernel);
    _MatrixArray<de::Half>* _dst = dynamic_cast<_MatrixArray<de::Half>*>(&dst);

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::hConv2_border_ignore_sk(_src, _kernel, _dst, &handle);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        decx::hConv2_border_zero_sk(_src, _kernel, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}



de::DH de::cuda::Conv2_multi_kernel(de::MatrixArray<float>& src, de::MatrixArray<float>& kernel, de::MatrixArray<float>& dst, const int flag)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    _MatrixArray<float>* _src = dynamic_cast<_MatrixArray<float>*>(&src);
    _MatrixArray<float>* _kernel = dynamic_cast<_MatrixArray<float>*>(&kernel);
    _MatrixArray<float>* _dst = dynamic_cast<_MatrixArray<float>*>(&dst);

    if (_src->ArrayNumber != _kernel->ArrayNumber) {
        decx::Matrix_number_not_matching(&handle);
        return handle;
    }

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::sConv2_border_ignore_mk(_src, _kernel, _dst, &handle);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        decx::sConv2_border_zero_mk(_src, _kernel, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}



de::DH de::cuda::Conv2_multi_kernel(de::MatrixArray<de::Half>& src, de::MatrixArray<de::Half>& kernel, de::MatrixArray<de::Half>& dst, const int flag)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    _MatrixArray<de::Half>* _src = dynamic_cast<_MatrixArray<de::Half>*>(&src);
    _MatrixArray<de::Half>* _kernel = dynamic_cast<_MatrixArray<de::Half>*>(&kernel);
    _MatrixArray<de::Half>* _dst = dynamic_cast<_MatrixArray<de::Half>*>(&dst);

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::hConv2_border_ignore_mk(_src, _kernel, _dst, &handle);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        decx::hConv2_border_zero_mk(_src, _kernel, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}



de::DH de::cuda::Conv2_single_kernel(de::GPU_MatrixArray<float>& src, de::GPU_Matrix<float>& kernel, de::GPU_MatrixArray<float>& dst, const int flag)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    _GPU_MatrixArray<float>* _src = dynamic_cast<_GPU_MatrixArray<float>*>(&src);
    _GPU_Matrix<float>* _kernel = dynamic_cast<_GPU_Matrix<float>*>(&kernel);
    _GPU_MatrixArray<float>* _dst = dynamic_cast<_GPU_MatrixArray<float>*>(&dst);

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::dev_sConv2_border_ignore_sk(_src, _kernel, _dst, &handle);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        decx::dev_sConv2_border_zero_sk(_src, _kernel, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}



de::DH de::cuda::Conv2_multi_kernel(de::GPU_MatrixArray<float>& src, de::GPU_MatrixArray<float>& kernel, de::GPU_MatrixArray<float>& dst, const int flag)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    _GPU_MatrixArray<float>* _src = dynamic_cast<_GPU_MatrixArray<float>*>(&src);
    _GPU_MatrixArray<float>* _kernel = dynamic_cast<_GPU_MatrixArray<float>*>(&kernel);
    _GPU_MatrixArray<float>* _dst = dynamic_cast<_GPU_MatrixArray<float>*>(&dst);

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::dev_sConv2_border_ignore_mk(_src, _kernel, _dst, &handle);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        decx::dev_sConv2_border_zero_mk(_src, _kernel, _dst, &handle);
        break;
    default:
        break;
    }
}




de::DH de::cuda::Conv2_single_kernel(de::GPU_MatrixArray<de::Half>& src, de::GPU_Matrix<de::Half>& kernel, de::GPU_MatrixArray<de::Half>& dst, const int flag)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    _GPU_MatrixArray<de::Half>* _src = dynamic_cast<_GPU_MatrixArray<de::Half>*>(&src);
    _GPU_Matrix<de::Half>* _kernel = dynamic_cast<_GPU_Matrix<de::Half>*>(&kernel);
    _GPU_MatrixArray<de::Half>* _dst = dynamic_cast<_GPU_MatrixArray<de::Half>*>(&dst);

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::dev_hConv2_border_ignore_sk(_src, _kernel, _dst, &handle);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        decx::dev_hConv2_border_zero_sk(_src, _kernel, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}



de::DH de::cuda::Conv2_multi_kernel(de::GPU_MatrixArray<de::Half>& src, de::GPU_MatrixArray<de::Half>& kernel, de::GPU_MatrixArray<de::Half>& dst, const int flag)
{
    de::DH handle;

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    _GPU_MatrixArray<de::Half>* _src = dynamic_cast<_GPU_MatrixArray<de::Half>*>(&src);
    _GPU_MatrixArray<de::Half>* _kernel = dynamic_cast<_GPU_MatrixArray<de::Half>*>(&kernel);
    _GPU_MatrixArray<de::Half>* _dst = dynamic_cast<_GPU_MatrixArray<de::Half>*>(&dst);

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::dev_hConv2_border_ignore_mk(_src, _kernel, _dst, &handle);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        decx::dev_hConv2_border_zero_mk(_src, _kernel, _dst, &handle);
        break;
    default:
        break;
    }
}



#endif        // #ifndef _CONV2_MC_H_