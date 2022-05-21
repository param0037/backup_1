/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once


#include "FFT1D_sub_funcs.h"



namespace de
{
    namespace fft
    {
        _DECX_API_ de::DH FFT1D_R2C_f(_VFLOAT_& src, _VCPF_& dst);


        _DECX_API_ de::DH FFT1D_C2C_f(_VCPF_& src, _VCPF_& dst);
    }
}




de::DH de::fft::FFT1D_R2C_f(_VFLOAT_& src, _VCPF_& dst)
{
    de::DH handle;

    decx::_Vector<float>* _src = dynamic_cast<decx::_Vector<float>*>(&src);
    decx::_Vector<de::CPf>* _dst = dynamic_cast<decx::_Vector<de::CPf>*>(&dst);

    if (cuP.is_init == false) {
        decx::Not_init(&handle);
        return handle;
    }

    const int src_len = _src->length;

    if (!decx::fft::check_apart(src_len)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::fft::FFT_Configs config;
    config.FFT1D_config_gen(src_len, &handle, S);

    _dst->re_construct(src_len, decx::DATA_STORE_TYPE::Page_Locked);

    if (config._is_2s) {    // == 1 : Mixradix; == 0 : Radix--2
        decx::fft::FFT1D_RC_b2_f(_src, _dst, &handle, &config, S);
    }
    else {
        decx::fft::FFT1D_R2C_MixBase_f(_src, _dst, &handle, &config, S);
        
    }
    config.release_inner_tmp();

    checkCudaErrors(cudaStreamDestroy(S));
    return handle;
}



de::DH de::fft::FFT1D_C2C_f(_VCPF_& src, _VCPF_& dst)
{
    de::DH handle;

    decx::_Vector<de::CPf>* _src = dynamic_cast<decx::_Vector<de::CPf>*>(&src);
    decx::_Vector<de::CPf>* _dst = dynamic_cast<decx::_Vector<de::CPf>*>(&dst);

    if (cuP.is_init == false) {
        decx::Not_init(&handle);
        return handle;
    }

    const int src_len = _src->length;
    if (!decx::fft::check_apart(src_len)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::fft::FFT_Configs config;
    config.FFT1D_config_gen(src_len, &handle, S);

    _dst->re_construct(src_len, decx::DATA_STORE_TYPE::Page_Locked);

    if (config._is_2s) {    // == 1 : Mixradix; == 0 : Radix--2
        decx::fft::FFT1D_CC_b2_f(_src, _dst, &handle, &config, S);
    }
    else {
        decx::fft::FFT1D_C2C_MixBase_f(_src, _dst, &handle, &config, S);
    }
    config.release_inner_tmp();

    checkCudaErrors(cudaStreamDestroy(S));
    return handle;
}