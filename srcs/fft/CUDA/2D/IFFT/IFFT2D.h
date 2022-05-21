/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#include "IFFT2D_sub_funcs.h"


namespace de
{
    namespace fft
    {
        _DECX_API_ de::DH IFFT2D_C2R_f(_CPF_& src, _FLOAT_& dst);


        _DECX_API_ de::DH IFFT2D_C2C_f(_CPF_& src, _CPF_& dst);
    }
}




de::DH de::fft::IFFT2D_C2R_f(_CPF_& src, _FLOAT_& dst)
{
    de::DH handle;

    decx::_Matrix<de::CPf>* _src = dynamic_cast<decx::_Matrix<de::CPf>*>(&src);
    decx::_Matrix<float>* _dst = dynamic_cast<decx::_Matrix<float>*>(&dst);

    if (cuP.is_init == false) {
        handle.error_type = decx::DECX_FAIL_not_init;
        handle.error_string = "CUDA should be initialize first";
        return handle;
    }

    const uint width = _src->width;
    const uint height = _src->height;

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::fft::FFT_Configs config_tmp;
    decx::fft::IFFT2D_C2R_organiser(_src, _dst, &handle, &config_tmp, S);

    checkCudaErrors(cudaStreamDestroy(S));
    return handle;
}




de::DH de::fft::IFFT2D_C2C_f(_CPF_& src, _CPF_& dst)
{
    de::DH handle;

    decx::_Matrix<de::CPf>* _src = dynamic_cast<decx::_Matrix<de::CPf>*>(&src);
    decx::_Matrix<de::CPf>* _dst = dynamic_cast<decx::_Matrix<de::CPf>*>(&dst);

    if (cuP.is_init == false) {
        handle.error_type = decx::DECX_FAIL_not_init;
        handle.error_string = "CUDA should be initialize first";
        return handle;
    }

    const uint width = _src->width;
    const uint height = _src->height;

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::fft::FFT_Configs config_tmp;
    decx::fft::IFFT2D_C2C_organiser(_src, _dst, &handle, &config_tmp, S);

    checkCudaErrors(cudaStreamDestroy(S));
    return handle;
}