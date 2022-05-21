/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#include "kernel.cuh"
#include "../../sort_and_chart.cuh"
#include "../../../../classes/core_types.h"


// Radix-2 FFT RC
namespace decx
{
    namespace fft
    {
        /**
        * The vector length will be in 2's power
        * @param src : input vector, containing data in type of float
        * @param dst : destinated vector, containing data in type of de::CPf
        */
        static void FFT2D_RC_b2_f(decx::_Matrix<float>* src, decx::_Matrix<de::CPf>* dst, de::DH* handle, decx::fft::FFT_Configs* config,
            cudaStream_t& S);



        /**
        * The vector length will be in 2's power
        * @param src : input vector, containing data in type of float
        * @param dst : destinated vector, containing data in type of de::CPf
        */
        static void FFT2D_CC_b2_f(decx::_Matrix<de::CPf>* src, decx::_Matrix<de::CPf>* dst, de::DH* handle, decx::fft::FFT_Configs* config,
            cudaStream_t& S);


        
        /**
        * Before being called, the bases and other parameters are all transferred to device memory, NO memcpy
        * op. in this function!
        * @param eq_rows : the number of small FFT fragments
        * @param eq_cols : the length of a FFT fragment
        * @param base_range : indicates the range of the bases numbers stored in constant memory
        */
        static void FFT2D_2N_HFkernel_caller_RC(const size_t                    src_len,
                                                decx::alloc::MIF<float2>*        dev_tmp1,
                                                decx::alloc::MIF<float2>*        dev_tmp2,
                                                decx::fft::FFT_Configs*            config,
                                                cudaStream_t&                    S);


        /**
        * Before being called, the bases and other parameters are all transferred to device memory, NO memcpy
        * op. in this function!
        * @param eq_rows : the number of small FFT fragments
        * @param eq_cols : the length of a FFT fragment
        * @param base_range : indicates the range of the bases numbers stored in constant memory
        */
        static void FFT2D_2N_HFkernel_caller_CC(const size_t                    src_len,
                                                decx::alloc::MIF<float2>*        dev_tmp1,
                                                decx::alloc::MIF<float2>*        dev_tmp2,
                                                decx::fft::FFT_Configs*            config,
                                                cudaStream_t&                    S);




        static void FFT2D_R2C_organiser(decx::_Matrix<float>*        src,
                                        decx::_Matrix<de::CPf>*        dst,
                                        de::DH*                        handle,
                                        decx::fft::FFT_Configs*        config,
                                        cudaStream_t&                S);



        static void FFT2D_C2C_organiser(decx::_Matrix<de::CPf>*        src,
                                        decx::_Matrix<de::CPf>*        dst,
                                        de::DH*                        handle,
                                        decx::fft::FFT_Configs*        config,
                                        cudaStream_t&                S);



        static void FFT2D_R2C_MixRadix_f_kernel_caller_halved(decx::alloc::MIF<float2>*        dev_tmp1,
                                                              decx::alloc::MIF<float2>*        dev_tmp2,
                                                              decx::fft::FFT_Configs*        config,
                                                              size_t                        _fragment, 
                                                              cudaStream_t&                    S);



        static void FFT2D_C2C_MixRadix_f_kernel_caller_halved(decx::alloc::MIF<float2>*        dev_tmp1,
                                                              decx::alloc::MIF<float2>*        dev_tmp2,
                                                              decx::fft::FFT_Configs*        config,
                                                              size_t                        _fragment, 
                                                              cudaStream_t&                    S);



        static void FFT2D_R2C_MixRadix_f_kernel_caller(decx::alloc::MIF<float2>*        dev_tmp1,
                                                       decx::alloc::MIF<float2>*        dev_tmp2, 
                                                       decx::fft::FFT_Configs*            config, 
                                                       size_t                            _fragment, 
                                                       cudaStream_t&                    S);



        static void FFT2D_C2C_MixRadix_f_kernel_caller(decx::alloc::MIF<float2>*        dev_tmp1,
                                                       decx::alloc::MIF<float2>*        dev_tmp2, 
                                                       decx::fft::FFT_Configs*            config, 
                                                       size_t                            _fragment, 
                                                       cudaStream_t&                    S);
    }
}




static 
void decx::fft::FFT2D_R2C_MixRadix_f_kernel_caller_halved(decx::alloc::MIF<float2>*        dev_tmp1,
                                                          decx::alloc::MIF<float2>*        dev_tmp2,
                                                          decx::fft::FFT_Configs*        config,
                                                          size_t                        _fragment, 
                                                          cudaStream_t&                    S)
{
    if (dev_tmp1->leading) {
        __cu_MixRadix_sort2D_R_halved << <config->_LP_FFT2D->Sort[0], config->_LP_FFT2D->Sort[1], 0, S >> > (
            dev_tmp1->mem, reinterpret_cast<float4*>(dev_tmp2->mem),
            _fragment, config->_base_num,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);
        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
    }
    else {
        __cu_MixRadix_sort2D_R_halved << <config->_LP_FFT2D->Sort[0], config->_LP_FFT2D->Sort[1], 0, S >> > (
            dev_tmp2->mem, reinterpret_cast<float4*>(dev_tmp1->mem),
            _fragment, config->_base_num,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);
        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
    }

    int warp_proc_domain = 1, warp_len = 1;
    int total_thr_num;
    for (int i = 0; i < config->_base.size(); ++i)
    {
        int current_base = config->_base.operator[](i);
        warp_proc_domain *= current_base;
        total_thr_num = _fragment / current_base;

        switch (current_base)
        {
        case 2:
            if (dev_tmp1->leading) {
                cu_FFT2D_b2_CC_halved << < config->_LP_FFT2D->Radix_2[0], config->_LP_FFT2D->Radix_2[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_FFT2D_b2_CC_halved << < config->_LP_FFT2D->Radix_2[0], config->_LP_FFT2D->Radix_2[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        case 3:
            if (dev_tmp1->leading) {
                cu_FFT2D_b3_CC_halved << < config->_LP_FFT2D->Radix_3[0], config->_LP_FFT2D->Radix_3[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_FFT2D_b3_CC_halved << < config->_LP_FFT2D->Radix_3[0], config->_LP_FFT2D->Radix_3[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        case 4:
            if (dev_tmp1->leading) {
                cu_FFT2D_b4_CC_halved << < config->_LP_FFT2D->Radix_4[0], config->_LP_FFT2D->Radix_4[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_FFT2D_b4_CC_halved << < config->_LP_FFT2D->Radix_4[0], config->_LP_FFT2D->Radix_4[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        case 5:
            if (dev_tmp1->leading) {
                cu_FFT2D_b5_CC_halved << < config->_LP_FFT2D->Radix_5[0], config->_LP_FFT2D->Radix_5[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_FFT2D_b5_CC_halved << < config->_LP_FFT2D->Radix_5[0], config->_LP_FFT2D->Radix_5[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;
        default:
            break;
        }

        warp_len *= current_base;
    }

    if (dev_tmp1->leading) {
        cu_FFT2D_b2_single_clamp_last << < config->_LP_FFT2D->Radix_2_last[0], config->_LP_FFT2D->Radix_2_last[1], 0, S >> > (
            dev_tmp1->mem, dev_tmp2->mem, _fragment * 2, _fragment, _fragment,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
    }
    else {
        cu_FFT2D_b2_single_clamp_last << < config->_LP_FFT2D->Radix_2_last[0], config->_LP_FFT2D->Radix_2_last[1], 0, S >> > (
            dev_tmp2->mem, dev_tmp1->mem, _fragment * 2, _fragment, _fragment,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
    }
}



static 
void decx::fft::FFT2D_C2C_MixRadix_f_kernel_caller_halved(decx::alloc::MIF<float2>*        dev_tmp1,
                                                          decx::alloc::MIF<float2>*        dev_tmp2,
                                                          decx::fft::FFT_Configs*        config,
                                                          size_t                        _fragment, 
                                                          cudaStream_t&                    S)
{
    if (dev_tmp1->leading) {
        __cu_MixRadix_sort2D_C_halved << <config->_LP_FFT2D->Sort[0], config->_LP_FFT2D->Sort[1], 0, S >> > (
            reinterpret_cast<float4*>(dev_tmp1->mem), reinterpret_cast<float4*>(dev_tmp2->mem),
            _fragment, config->_base_num,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);
        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
    }
    else {
        __cu_MixRadix_sort2D_C_halved << <config->_LP_FFT2D->Sort[0], config->_LP_FFT2D->Sort[1], 0, S >> > (
            reinterpret_cast<float4*>(dev_tmp2->mem), reinterpret_cast<float4*>(dev_tmp1->mem),
            _fragment, config->_base_num,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);
        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
    }

    int warp_proc_domain = 1, warp_len = 1;
    int total_thr_num;
    for (int i = 0; i < config->_base.size(); ++i)
    {
        int current_base = config->_base.operator[](i);
        warp_proc_domain *= current_base;
        total_thr_num = _fragment / current_base;

        switch (current_base)
        {
        case 2:
            if (dev_tmp1->leading) {
                cu_FFT2D_b2_CC_halved << < config->_LP_FFT2D->Radix_2[0], config->_LP_FFT2D->Radix_2[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_FFT2D_b2_CC_halved << < config->_LP_FFT2D->Radix_2[0], config->_LP_FFT2D->Radix_2[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        case 3:
            if (dev_tmp1->leading) {
                cu_FFT2D_b3_CC_halved << < config->_LP_FFT2D->Radix_3[0], config->_LP_FFT2D->Radix_3[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_FFT2D_b3_CC_halved << < config->_LP_FFT2D->Radix_3[0], config->_LP_FFT2D->Radix_3[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        case 4:
            if (dev_tmp1->leading) {
                cu_FFT2D_b4_CC_halved << < config->_LP_FFT2D->Radix_4[0], config->_LP_FFT2D->Radix_4[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_FFT2D_b4_CC_halved << < config->_LP_FFT2D->Radix_4[0], config->_LP_FFT2D->Radix_4[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        case 5:
            if (dev_tmp1->leading) {
                cu_FFT2D_b5_CC_halved << < config->_LP_FFT2D->Radix_5[0], config->_LP_FFT2D->Radix_5[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_FFT2D_b5_CC_halved << < config->_LP_FFT2D->Radix_5[0], config->_LP_FFT2D->Radix_5[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;
        default:
            break;
        }

        warp_len *= current_base;
    }

    if (dev_tmp1->leading) {
        cu_FFT2D_b2_single_clamp_last << < config->_LP_FFT2D->Radix_2_last[0], config->_LP_FFT2D->Radix_2_last[1], 0, S >> > (
            dev_tmp1->mem, dev_tmp2->mem, _fragment * 2, _fragment, _fragment,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
    }
    else {
        cu_FFT2D_b2_single_clamp_last << < config->_LP_FFT2D->Radix_2_last[0], config->_LP_FFT2D->Radix_2_last[1], 0, S >> > (
            dev_tmp2->mem, dev_tmp1->mem, _fragment * 2, _fragment, _fragment,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
    }
}




static void decx::fft::FFT2D_R2C_MixRadix_f_kernel_caller(decx::alloc::MIF<float2>*            dev_tmp1,
                                                          decx::alloc::MIF<float2>*            dev_tmp2, 
                                                          decx::fft::FFT_Configs*            config, 
                                                          size_t                            _fragment, 
                                                          cudaStream_t&                        S)
{
    if (dev_tmp1->leading) {
        __cu_MixRadix_sort2D_R << <config->_LP_FFT2D->Sort[0], config->_LP_FFT2D->Sort[1], 0, S >> > (
            reinterpret_cast<float*>(dev_tmp1->mem), dev_tmp2->mem, _fragment, config->_base_num,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);
        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
    }
    else {
        __cu_MixRadix_sort2D_R << <config->_LP_FFT2D->Sort[0], config->_LP_FFT2D->Sort[1], 0, S >> > (
            reinterpret_cast<float*>(dev_tmp2->mem), dev_tmp1->mem, _fragment, config->_base_num,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);
        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
    }

    int warp_proc_domain = 1, warp_len = 1;
    int total_thr_num;
    for (int i = 0; i < config->_base.size(); ++i)
    {
        int current_base = config->_base.operator[](i);
        warp_proc_domain *= current_base;
        total_thr_num = _fragment / current_base;

        switch (current_base)
        {
        case 3:
            if (i != config->_base.size() - 1) {
                if (dev_tmp1->leading) {
                    cu_FFT2D_b3_CC << < config->_LP_FFT2D->Radix_3[0], config->_LP_FFT2D->Radix_3[1], 0, S >> > (
                        dev_tmp1->mem, dev_tmp2->mem,
                        total_thr_num, warp_proc_domain, warp_len,
                        config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

                    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
                }
                else {
                    cu_FFT2D_b3_CC << < config->_LP_FFT2D->Radix_3[0], config->_LP_FFT2D->Radix_3[1], 0, S >> > (
                        dev_tmp2->mem, dev_tmp1->mem,
                        total_thr_num, warp_proc_domain, warp_len,
                        config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

                    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
                }
            }
            else {
                if (dev_tmp1->leading) {
                    cu_FFT2D_b3_CC_trans << < config->_LP_FFT2D->Radix_3[0], config->_LP_FFT2D->Radix_3[1], 0, S >> > (
                        dev_tmp1->mem, dev_tmp2->mem,
                        total_thr_num, warp_proc_domain, warp_len,
                        config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

                    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
                }
                else {
                    cu_FFT2D_b3_CC_trans << < config->_LP_FFT2D->Radix_3[0], config->_LP_FFT2D->Radix_3[1], 0, S >> > (
                        dev_tmp2->mem, dev_tmp1->mem,
                        total_thr_num, warp_proc_domain, warp_len,
                        config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

                    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
                }
            }
            break;

        case 5:
            if (i != config->_base.size() - 1) {
                if (dev_tmp1->leading) {
                    cu_FFT2D_b5_CC << < config->_LP_FFT2D->Radix_5[0], config->_LP_FFT2D->Radix_5[1], 0, S >> > (
                        dev_tmp1->mem, dev_tmp2->mem,
                        total_thr_num, warp_proc_domain, warp_len,
                        config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

                    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
                }
                else {
                    cu_FFT2D_b5_CC << < config->_LP_FFT2D->Radix_5[0], config->_LP_FFT2D->Radix_5[1], 0, S >> > (
                        dev_tmp2->mem, dev_tmp1->mem,
                        total_thr_num, warp_proc_domain, warp_len,
                        config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

                    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
                }
            }
            else {
                if (dev_tmp1->leading) {
                    cu_FFT2D_b5_CC_trans << < config->_LP_FFT2D->Radix_5[0], config->_LP_FFT2D->Radix_5[1], 0, S >> > (
                        dev_tmp1->mem, dev_tmp2->mem,
                        total_thr_num, warp_proc_domain, warp_len,
                        config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

                    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
                }
                else {
                    cu_FFT2D_b5_CC_trans << < config->_LP_FFT2D->Radix_5[0], config->_LP_FFT2D->Radix_5[1], 0, S >> > (
                        dev_tmp2->mem, dev_tmp1->mem,
                        total_thr_num, warp_proc_domain, warp_len,
                        config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

                    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
                }
            }
            break;
        default:
            break;
        }

        warp_len *= current_base;
    }
}




static void decx::fft::FFT2D_C2C_MixRadix_f_kernel_caller(decx::alloc::MIF<float2>*            dev_tmp1,
                                                          decx::alloc::MIF<float2>*            dev_tmp2, 
                                                          decx::fft::FFT_Configs*            config, 
                                                          size_t                            _fragment, 
                                                          cudaStream_t&                        S)
{
    if (dev_tmp1->leading) {
        __cu_MixRadix_sort2D_C << <config->_LP_FFT2D->Sort[0], config->_LP_FFT2D->Sort[1], 0, S >> > (
            dev_tmp1->mem, dev_tmp2->mem, _fragment, config->_base_num,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);
        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
    }
    else {
        __cu_MixRadix_sort2D_C << <config->_LP_FFT2D->Sort[0], config->_LP_FFT2D->Sort[1], 0, S >> > (
            dev_tmp2->mem, dev_tmp1->mem, _fragment, config->_base_num,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);
        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
    }

    int warp_proc_domain = 1, warp_len = 1;
    int total_thr_num;
    for (int i = 0; i < config->_base.size(); ++i)
    {
        int current_base = config->_base.operator[](i);
        warp_proc_domain *= current_base;
        total_thr_num = _fragment / current_base;

        switch (current_base)
        {
        case 3:
            if (i != config->_base.size() - 1) {
                if (dev_tmp1->leading) {
                    cu_FFT2D_b3_CC << < config->_LP_FFT2D->Radix_3[0], config->_LP_FFT2D->Radix_3[1], 0, S >> > (
                        dev_tmp1->mem, dev_tmp2->mem,
                        total_thr_num, warp_proc_domain, warp_len,
                        config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

                    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
                }
                else {
                    cu_FFT2D_b3_CC << < config->_LP_FFT2D->Radix_3[0], config->_LP_FFT2D->Radix_3[1], 0, S >> > (
                        dev_tmp2->mem, dev_tmp1->mem,
                        total_thr_num, warp_proc_domain, warp_len,
                        config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

                    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
                }
            }
            else {
                if (dev_tmp1->leading) {
                    cu_FFT2D_b3_CC_trans << < config->_LP_FFT2D->Radix_3[0], config->_LP_FFT2D->Radix_3[1], 0, S >> > (
                        dev_tmp1->mem, dev_tmp2->mem,
                        total_thr_num, warp_proc_domain, warp_len,
                        config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

                    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
                }
                else {
                    cu_FFT2D_b3_CC_trans << < config->_LP_FFT2D->Radix_3[0], config->_LP_FFT2D->Radix_3[1], 0, S >> > (
                        dev_tmp2->mem, dev_tmp1->mem,
                        total_thr_num, warp_proc_domain, warp_len,
                        config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

                    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
                }
            }
            break;

        case 5:
            if (i != config->_base.size() - 1) {
                if (dev_tmp1->leading) {
                    cu_FFT2D_b5_CC << < config->_LP_FFT2D->Radix_5[0], config->_LP_FFT2D->Radix_5[1], 0, S >> > (
                        dev_tmp1->mem, dev_tmp2->mem,
                        total_thr_num, warp_proc_domain, warp_len,
                        config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

                    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
                }
                else {
                    cu_FFT2D_b5_CC << < config->_LP_FFT2D->Radix_5[0], config->_LP_FFT2D->Radix_5[1], 0, S >> > (
                        dev_tmp2->mem, dev_tmp1->mem,
                        total_thr_num, warp_proc_domain, warp_len,
                        config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

                    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
                }
            }
            else {
                if (dev_tmp1->leading) {
                    cu_FFT2D_b5_CC_trans << < config->_LP_FFT2D->Radix_5[0], config->_LP_FFT2D->Radix_5[1], 0, S >> > (
                        dev_tmp1->mem, dev_tmp2->mem,
                        total_thr_num, warp_proc_domain, warp_len,
                        config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

                    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
                }
                else {
                    cu_FFT2D_b5_CC_trans << < config->_LP_FFT2D->Radix_5[0], config->_LP_FFT2D->Radix_5[1], 0, S >> > (
                        dev_tmp2->mem, dev_tmp1->mem,
                        total_thr_num, warp_proc_domain, warp_len,
                        config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

                    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
                }
            }
            break;
        default:
            break;
        }

        warp_len *= current_base;
    }
}



static void decx::fft::FFT2D_2N_HFkernel_caller_RC(const size_t                        _fragment,
                                                   decx::alloc::MIF<float2>*        dev_tmp1,
                                                   decx::alloc::MIF<float2>*        dev_tmp2,
                                                   decx::fft::FFT_Configs*            config,
                                                   cudaStream_t&                    S)
{
    int consec_frag = 1 << (2 * config->_consecutive_4);
    
    if (dev_tmp1->leading) {
        __cu_MixRadix_sort2D_R_halved << <config->_LP_FFT2D->Sort[0], config->_LP_FFT2D->Sort[1], 0, S >> > (
            dev_tmp1->mem, reinterpret_cast<float4*>(dev_tmp2->mem), _fragment, config->_base_num,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

        cu_FFT2D_b4_CC_consec128_halved << <config->_LP_FFT2D->Radix_4_consec[0], config->_LP_FFT2D->Radix_4_consec[1], 0, S >> > (
            reinterpret_cast<float4*>(dev_tmp2->mem),
            reinterpret_cast<float4*>(dev_tmp1->mem),
            consec_frag / 4, config->_consecutive_4,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
    }
    else {
        __cu_MixRadix_sort2D_R_halved << <config->_LP_FFT2D->Sort[0], config->_LP_FFT2D->Sort[1], 0, S >> > (
            dev_tmp2->mem, reinterpret_cast<float4*>(dev_tmp1->mem), _fragment, config->_base_num,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

        cu_FFT2D_b4_CC_consec128_halved << <config->_LP_FFT2D->Radix_4_consec[0], config->_LP_FFT2D->Radix_4_consec[1], 0, S >> > (
            reinterpret_cast<float4*>(dev_tmp1->mem),
            reinterpret_cast<float4*>(dev_tmp2->mem),
            consec_frag / 4, config->_consecutive_4,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
    }

    int warp_proc_domain = consec_frag, warp_len = consec_frag;
    int total_thr_num;
    for (int i = config->_consecutive_4; i < config->_base_num; ++i)
    {
        int current_base = config->_base.operator[](i);
        warp_proc_domain *= current_base;
        total_thr_num = _fragment / current_base;
        switch (current_base)
        {
        case 2:
            if (dev_tmp1->leading) {
                cu_FFT2D_b2_CC_halved << < config->_LP_FFT2D->Radix_2[0], config->_LP_FFT2D->Radix_2[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_FFT2D_b2_CC_halved << < config->_LP_FFT2D->Radix_2[0], config->_LP_FFT2D->Radix_2[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        case 4:
            if (dev_tmp1->leading) {
                cu_FFT2D_b4_CC_halved << < config->_LP_FFT2D->Radix_4[0], config->_LP_FFT2D->Radix_4[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_FFT2D_b4_CC_halved << < config->_LP_FFT2D->Radix_4[0], config->_LP_FFT2D->Radix_4[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        default:
            break;
        }

        warp_len *= current_base;
    }

    if (dev_tmp1->leading) {
        cu_FFT2D_b2_single_clamp_last << < config->_LP_FFT2D->Radix_2_last[0], config->_LP_FFT2D->Radix_2_last[1], 0, S >> > (
            dev_tmp1->mem, dev_tmp2->mem, _fragment * 2, _fragment, _fragment,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
    }
    else {
        cu_FFT2D_b2_single_clamp_last << < config->_LP_FFT2D->Radix_2_last[0], config->_LP_FFT2D->Radix_2_last[1], 0, S >> > (
            dev_tmp2->mem, dev_tmp1->mem, _fragment * 2, _fragment, _fragment,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
    }
}



static void decx::fft::FFT2D_2N_HFkernel_caller_CC(const size_t                        _fragment,
                                                   decx::alloc::MIF<float2>*        dev_tmp1,
                                                   decx::alloc::MIF<float2>*        dev_tmp2,
                                                   decx::fft::FFT_Configs*            config,
                                                   cudaStream_t&                    S)
{
    int consec_frag = 1 << (2 * config->_consecutive_4);

    if (dev_tmp1->leading) {
        __cu_MixRadix_sort2D_C_halved << <config->_LP_FFT2D->Sort[0], config->_LP_FFT2D->Sort[1], 0, S >> > (
            reinterpret_cast<float4*>(dev_tmp1->mem), reinterpret_cast<float4*>(dev_tmp2->mem), 
            _fragment, config->_base_num,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

        cu_FFT2D_b4_CC_consec128_halved << <config->_LP_FFT2D->Radix_4_consec[0], config->_LP_FFT2D->Radix_4_consec[1], 0, S >> > (
            reinterpret_cast<float4*>(dev_tmp2->mem),
            reinterpret_cast<float4*>(dev_tmp1->mem),
            consec_frag / 4, config->_consecutive_4,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
    }
    else {
        __cu_MixRadix_sort2D_C_halved << <config->_LP_FFT2D->Sort[0], config->_LP_FFT2D->Sort[1], 0, S >> > (
            reinterpret_cast<float4*>(dev_tmp2->mem), reinterpret_cast<float4*>(dev_tmp1->mem), 
            _fragment, config->_base_num,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

        cu_FFT2D_b4_CC_consec128_halved << <config->_LP_FFT2D->Radix_4_consec[0], config->_LP_FFT2D->Radix_4_consec[1], 0, S >> > (
            reinterpret_cast<float4*>(dev_tmp1->mem),
            reinterpret_cast<float4*>(dev_tmp2->mem),
            consec_frag / 4, config->_consecutive_4,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
    }

    int warp_proc_domain = consec_frag, warp_len = consec_frag;
    int total_thr_num;
    for (int i = config->_consecutive_4; i < config->_base_num; ++i)
    {
        int current_base = config->_base.operator[](i);
        warp_proc_domain *= current_base;
        total_thr_num = _fragment / current_base;
        switch (current_base)
        {
        case 2:
            if (dev_tmp1->leading) {
                cu_FFT2D_b2_CC_halved << < config->_LP_FFT2D->Radix_2[0], config->_LP_FFT2D->Radix_2[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_FFT2D_b2_CC_halved << < config->_LP_FFT2D->Radix_2[0], config->_LP_FFT2D->Radix_2[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        case 4:
            if (dev_tmp1->leading) {
                cu_FFT2D_b4_CC_halved << < config->_LP_FFT2D->Radix_4[0], config->_LP_FFT2D->Radix_4[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_FFT2D_b4_CC_halved << < config->_LP_FFT2D->Radix_4[0], config->_LP_FFT2D->Radix_4[1], 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len,
                    config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc / 2);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        default:
            break;
        }

        warp_len *= current_base;
    }

    if (dev_tmp1->leading) {
        cu_FFT2D_b2_single_clamp_last << < config->_LP_FFT2D->Radix_2_last[0], config->_LP_FFT2D->Radix_2_last[1], 0, S >> > (
            dev_tmp1->mem, dev_tmp2->mem, _fragment * 2, _fragment, _fragment,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
    }
    else {
        cu_FFT2D_b2_single_clamp_last << < config->_LP_FFT2D->Radix_2_last[0], config->_LP_FFT2D->Radix_2_last[1], 0, S >> > (
            dev_tmp2->mem, dev_tmp1->mem, _fragment * 2, _fragment, _fragment,
            config->_LP_FFT2D->_Hsrc, config->_LP_FFT2D->_Wsrc);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
    }
}




static void decx::fft::FFT2D_R2C_organiser(decx::_Matrix<float>*        src,
                                           decx::_Matrix<de::CPf>*        dst,
                                           de::DH*                        handle,
                                           decx::fft::FFT_Configs*        config,
                                           cudaStream_t&                S)
{
    const int _width = src->width;
    const int _height = src->height;

    // fft along rows
    if (!decx::fft::check_apart(_width)) {
        decx::err::FFT_Error_length(handle);
        Print_Error_Message(4, FFT_ERROR_WIDTH);
        return;
    }
    if (!decx::fft::check_apart(_height)) {
        decx::err::FFT_Error_length(handle);
        Print_Error_Message(4, FFT_ERROR_HEIGHT);
        return;
    }
    config->FFT1D_config_gen(_width, handle, S);
    config->FFT2D_launch_param_gen(_width, _height);
    
    dst->re_construct(_width, _height, decx::DATA_STORE_TYPE::Page_Locked);

    decx::PtrInfo<de::CPf> dev_tmp;
    decx::alloc::MIF<float2> dev_tmp1, dev_tmp2;

    const size_t req_size = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    if (decx::alloc::_device_malloc<de::CPf>(&dev_tmp, 2 * req_size * sizeof(de::CPf))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    dev_tmp1.mem = reinterpret_cast<float2*>(dev_tmp.ptr);
    dev_tmp2.mem = reinterpret_cast<float2*>(dev_tmp1.mem + req_size);

    checkCudaErrors(cudaMemcpy2DAsync(
        dev_tmp1.mem, _width * sizeof(float), src->Mat.ptr, src->pitch * sizeof(float), _width * sizeof(float), _height,
        cudaMemcpyHostToDevice, S));
    decx::utils::set_mutex_memory_state<float2, float2>(&dev_tmp1, &dev_tmp2);
    
    if (config->_is_2s) {    // it can definitely be divided into integer by 2
        const int _fragment = _width / 2;
        decx::fft::FFT2D_2N_HFkernel_caller_RC(_fragment, &dev_tmp1, &dev_tmp2, config, S);
    }
    else {
        if (config->_can_halve) {
            const int _fragment = _width / 2;
            decx::fft::FFT2D_R2C_MixRadix_f_kernel_caller_halved(&dev_tmp1, &dev_tmp2, config, _fragment, S);
        }
        else {
            decx::fft::FFT2D_R2C_MixRadix_f_kernel_caller(&dev_tmp1, &dev_tmp2, config, _width, S);
        }
    }
    
    // fft along cols
    config->FFT1D_config_gen(_height, handle, S);
    config->FFT2D_launch_param_gen(_height, _width);

    if (config->_is_2s) {    // it can definitely be divided into integer by 2
        const int _fragment = _height / 2;
        decx::fft::FFT2D_2N_HFkernel_caller_CC(_fragment, &dev_tmp1, &dev_tmp2, config, S);
    }
    else {
        if (config->_can_halve) {
            const int _fragment = _height / 2;
            decx::fft::FFT2D_C2C_MixRadix_f_kernel_caller_halved(&dev_tmp1, &dev_tmp2, config, _fragment, S);
        }
        else {
            decx::fft::FFT2D_C2C_MixRadix_f_kernel_caller(&dev_tmp1, &dev_tmp2, config, _height, S);
        }
    }
    
    if (dev_tmp1.leading) {
        checkCudaErrors(cudaMemcpy2DAsync(
            dst->Mat.ptr, dst->pitch * sizeof(de::CPf), dev_tmp1.mem, _width * sizeof(de::CPf), _width * sizeof(de::CPf), _height,
            cudaMemcpyDeviceToHost, S));
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(
            dst->Mat.ptr, dst->pitch * sizeof(de::CPf), dev_tmp2.mem, _width * sizeof(de::CPf), _width * sizeof(de::CPf), _height,
            cudaMemcpyDeviceToHost, S));
    }

    checkCudaErrors(cudaDeviceSynchronize());
}





static void decx::fft::FFT2D_C2C_organiser(decx::_Matrix<de::CPf>*        src,
                                           decx::_Matrix<de::CPf>*        dst,
                                           de::DH*                        handle,
                                           decx::fft::FFT_Configs*        config,
                                           cudaStream_t&                  S)
{
    const int _width = src->width;
    const int _height = src->height;

    if (!decx::fft::check_apart(_width)) {
        decx::err::FFT_Error_length(handle);
        Print_Error_Message(4, FFT_ERROR_WIDTH);
        return;
    }
    if (!decx::fft::check_apart(_height)) {
        decx::err::FFT_Error_length(handle);
        Print_Error_Message(4, FFT_ERROR_HEIGHT);
        return;
    }

    decx::PtrInfo<de::CPf> dev_tmp;
    decx::alloc::MIF<float2> dev_tmp1, dev_tmp2;

    dst->re_construct(_width, _height, decx::DATA_STORE_TYPE::Page_Locked);

    const size_t req_size = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    if (decx::alloc::_device_malloc<de::CPf>(&dev_tmp, 2 * req_size * sizeof(de::CPf))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    dev_tmp1.mem = reinterpret_cast<float2*>(dev_tmp.ptr);
    dev_tmp2.mem = reinterpret_cast<float2*>(dev_tmp1.mem + req_size);

    checkCudaErrors(cudaMemcpy2DAsync(
        dev_tmp1.mem, _width * sizeof(de::CPf), src->Mat.ptr, src->pitch * sizeof(de::CPf), _width * sizeof(de::CPf), _height,
        cudaMemcpyHostToDevice, S));
    decx::utils::set_mutex_memory_state<float2, float2>(&dev_tmp1, &dev_tmp2);
    
    // fft along rows
    config->FFT1D_config_gen(_width, handle, S);
    config->FFT2D_launch_param_gen(_width, _height);

    if (config->_is_2s) {    // it can definitely be divided into integer by 2
        const int _fragment = _width / 2;
        decx::fft::FFT2D_2N_HFkernel_caller_CC(_fragment, &dev_tmp1, &dev_tmp2, config, S);
    }
    else {
        if (config->_can_halve) {
            const int _fragment = _width / 2;
            decx::fft::FFT2D_C2C_MixRadix_f_kernel_caller_halved(&dev_tmp1, &dev_tmp2, config, _fragment, S);
        }
        else {
            decx::fft::FFT2D_C2C_MixRadix_f_kernel_caller(&dev_tmp1, &dev_tmp2, config, _width, S);
        }
    }
    
    // fft along cols
    config->FFT1D_config_gen(_height, handle, S);
    config->FFT2D_launch_param_gen(_height, _width);

    if (config->_is_2s) {    // it can definitely be divided into integer by 2
        const int _fragment = _height / 2;
        decx::fft::FFT2D_2N_HFkernel_caller_CC(_fragment, &dev_tmp1, &dev_tmp2, config, S);
    }
    else {
        if (config->_can_halve) {
            const int _fragment = _height / 2;
            decx::fft::FFT2D_C2C_MixRadix_f_kernel_caller_halved(&dev_tmp1, &dev_tmp2, config, _fragment, S);
        }
        else {
            decx::fft::FFT2D_C2C_MixRadix_f_kernel_caller(&dev_tmp1, &dev_tmp2, config, _height, S);
        }
    }

    if (dev_tmp1.leading) {
        checkCudaErrors(cudaMemcpy2DAsync(
            dst->Mat.ptr, dst->pitch * sizeof(de::CPf), dev_tmp1.mem, _width * sizeof(de::CPf), _width * sizeof(de::CPf), _height,
            cudaMemcpyDeviceToHost, S));
    }
    else {
        checkCudaErrors(cudaMemcpy2DAsync(
            dst->Mat.ptr, dst->pitch * sizeof(de::CPf), dev_tmp2.mem, _width * sizeof(de::CPf), _width * sizeof(de::CPf), _height,
            cudaMemcpyDeviceToHost, S));
    }

    checkCudaErrors(cudaDeviceSynchronize());
}