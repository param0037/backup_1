/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _IFFT1D_H_
#define _IFFT1D_H_


#include "IFFT1D_sub_funcs.h"



namespace de
{
    namespace fft
    {
        _DECX_API_ de::DH IFFT1D_C2R_f(_VCPF_& src, _VFLOAT_& dst);


        _DECX_API_ de::DH IFFT1D_C2C_f(_VCPF_& src, _VCPF_& dst);
    }
}




de::DH de::fft::IFFT1D_C2R_f(_VCPF_& src, _VFLOAT_& dst)
{
    de::DH handle;

    decx::_Vector<de::CPf>* _src = dynamic_cast<decx::_Vector<de::CPf>*>(&src);
    decx::_Vector<float>* _dst = dynamic_cast<decx::_Vector<float>*>(&dst);
    
    if (cuP.is_init == false) {
        decx::Not_init(&handle);
        return handle;
    }

    const int src_len = _src->length;
    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::fft::FFT_Configs config;
    config.FFT1D_config_gen(src_len, &handle, S);
    Error_Handle(handle, ., handle);

    if (config._is_2s) {    // == 1 : Mixradix; == 0 : Radix--2
        decx::fft::IFFT1D_CR_b2_f(_src, _dst, &handle, &config, S);
    }
    else {
        decx::fft::IFFT1D_C2R_MixBase_f(_src, _dst, &handle, &config, S);

    }
    config.release_inner_tmp();

    return handle;
}





de::DH de::fft::IFFT1D_C2C_f(_VCPF_& src, _VCPF_& dst)
{
    de::DH handle;

    decx::_Vector<de::CPf>* _src = dynamic_cast<decx::_Vector<de::CPf>*>(&src);
    decx::_Vector<de::CPf>* _dst = dynamic_cast<decx::_Vector<de::CPf>*>(&dst);

    if (cuP.is_init == false) {
        handle.error_type = decx::DECX_FAIL_not_init;
        handle.error_string = "CUDA should be initialize first";
        return handle;
    }

    const int src_len = _src->length;
    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::fft::FFT_Configs config;
    config.FFT1D_config_gen(src_len, &handle, S);

    if (config._is_2s) {    // == 1 : Mixradix; == 0 : Radix--2
        decx::fft::IFFT1D_CC_b2_f(_src, _dst, &handle, &config, S);
    }
    else {
        decx::fft::IFFT1D_C2C_MixBase_f(_src, _dst, &handle, &config, S);
    }
    config.release_inner_tmp();

    return handle;
}


#endif      // #ifndef _IFFT1D_H_