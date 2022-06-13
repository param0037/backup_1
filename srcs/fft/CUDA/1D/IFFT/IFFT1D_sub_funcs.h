/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _IFFT1D_SUB_FUNCS_H_
#define _IFFT1D_SUB_FUNCS_H_


#include "../../../../classes/Vector.h"
#include "kernel.cuh"
#include "../../sort_and_chart.cuh"
#include "../../../../classes/core_types.h"


// Radix-2 FFT RC
namespace decx
{
    /*namespace utils
    {
        template <typename _Ty1, typename _Ty2>
        inline void set_mutex_memory_state(decx::alloc::MIF<_Ty1>* _set_leading, decx::alloc::MIF<_Ty2>* _set_lagging);
    }*/

    namespace fft
    {
        /**
        * The vector length will be in 2's power
        * @param src : input vector, containing data in type of float
        * @param dst : destinated vector, containing data in type of de::CPf
        */
        static void IFFT1D_CR_b2_f(decx::_Vector<de::CPf>* src, decx::_Vector<float>* dst, de::DH* handle, decx::fft::FFT_Configs* config,
            cudaStream_t& S);



        /**
        * The vector length will be in 2's power
        * @param src : input vector, containing data in type of float
        * @param dst : destinated vector, containing data in type of de::CPf
        */
        static void IFFT1D_CC_b2_f(decx::_Vector<de::CPf>* src, decx::_Vector<de::CPf>* dst, de::DH* handle, decx::fft::FFT_Configs* config,
            cudaStream_t& S);


        
        /**
        * Before being called, the bases and other parameters are all transferred to device memory, NO memcpy
        * op. in this function!
        * @param eq_rows : the number of small FFT fragments
        * @param eq_cols : the length of a FFT fragment
        * @param base_range : indicates the range of the bases numbers stored in constant memory
        */
        static void IFFT1D_2N_HFkernel_caller_C2R(const size_t                    src_len,
                                                  decx::alloc::MIF<float2>*        dev_tmp1,
                                                  decx::alloc::MIF<float2>*        dev_tmp2,
                                                  decx::fft::FFT_Configs*        config,
                                                  cudaStream_t&                    S);



        /**
        * Before being called, the bases and other parameters are all transferred to device memory, NO memcpy
        * op. in this function!
        * @param eq_rows : the number of small FFT fragments
        * @param eq_cols : the length of a FFT fragment
        * @param base_range : indicates the range of the bases numbers stored in constant memory
        */
        static void IFFT1D_2N_HFkernel_caller_C2C(const size_t                    src_len,
                                                  decx::alloc::MIF<float2>*        dev_tmp1,
                                                  decx::alloc::MIF<float2>*        dev_tmp2,
                                                  decx::fft::FFT_Configs*        config,
                                                  cudaStream_t&                    S);



        static void IFFT1D_C2R_MixBase_f(decx::_Vector<de::CPf>*                src,
                                         decx::_Vector<float>*                    dst,
                                         de::DH*                                handle,
                                         decx::fft::FFT_Configs*                config,
                                         cudaStream_t&                            S);



        static void IFFT1D_C2C_MixBase_f(decx::_Vector<de::CPf>*                src,
                                         decx::_Vector<de::CPf>*                dst,
                                         de::DH*                                handle,
                                         decx::fft::FFT_Configs*                config,
                                         cudaStream_t&                            S);




        static void IFFT1D_C2R_MixRadix_f_kernel_caller_halved(decx::alloc::MIF<float2>*        dev_tmp1,
                                                              decx::alloc::MIF<float2>*            dev_tmp2,
                                                              decx::fft::FFT_Configs*            config,
                                                              size_t                            _fragment, 
                                                              cudaStream_t&                        S);



        static void IFFT1D_C2C_MixRadix_f_kernel_caller_halved(decx::alloc::MIF<float2>*        dev_tmp1,
                                                              decx::alloc::MIF<float2>*            dev_tmp2,
                                                              decx::fft::FFT_Configs*            config,
                                                              size_t                            _fragment, 
                                                              cudaStream_t&                        S);




        static void IFFT1D_C2R_MixRadix_f_kernel_caller(decx::alloc::MIF<float2>*        dev_tmp1,
                                                       decx::alloc::MIF<float2>*        dev_tmp2, 
                                                       decx::fft::FFT_Configs*            config, 
                                                       size_t                            _fragment, 
                                                       cudaStream_t&                    S);



        static void IFFT1D_C2C_MixRadix_f_kernel_caller(decx::alloc::MIF<float2>*        dev_tmp1,
                                                       decx::alloc::MIF<float2>*        dev_tmp2, 
                                                       decx::fft::FFT_Configs*            config, 
                                                       size_t                            _fragment, 
                                                       cudaStream_t&                    S);
    }
}



#define IFFT_KERNEL_FRAME_C2R(_kernel_main, _kernel_last){    \
if (i == config->_base.size() - 1) {    \
    if (dev_tmp1->leading) {    \
        _kernel_last << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (    \
            dev_tmp1->mem, reinterpret_cast<float*>(dev_tmp2->mem),    total_thr_num, warp_proc_domain, warp_len);    \
        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);    \
    }    \
    else {    \
        _kernel_last << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (    \
            dev_tmp2->mem, reinterpret_cast<float*>(dev_tmp1->mem),    total_thr_num, warp_proc_domain, warp_len);    \
        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);    \
    }    \
}    \
else {    \
    if (dev_tmp1->leading) {    \
        _kernel_main << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (    \
            dev_tmp1->mem, dev_tmp2->mem, total_thr_num, warp_proc_domain, warp_len);    \
        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);    \
    }    \
    else {    \
        _kernel_main << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (    \
            dev_tmp2->mem, dev_tmp1->mem, total_thr_num, warp_proc_domain, warp_len);    \
        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);    \
    }    \
}    \
}




#define IFFT_KERNEL_FRAME_C2C(_kernel_main, _kernel_last){    \
if (i == config->_base.size() - 1) {    \
    if (dev_tmp1->leading) {    \
        _kernel_last << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (    \
            dev_tmp1->mem, dev_tmp2->mem, total_thr_num, warp_proc_domain, warp_len);    \
        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);    \
    }    \
    else {    \
        _kernel_last << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (    \
            dev_tmp2->mem, dev_tmp1->mem, total_thr_num, warp_proc_domain, warp_len);    \
        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);    \
    }    \
}    \
else {    \
    if (dev_tmp1->leading) {    \
        _kernel_main << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (    \
            dev_tmp1->mem, dev_tmp2->mem, total_thr_num, warp_proc_domain, warp_len);    \
        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);    \
    }    \
    else {    \
        _kernel_main << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (    \
            dev_tmp2->mem, dev_tmp1->mem, total_thr_num, warp_proc_domain, warp_len);    \
        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);    \
    }    \
}    \
}





static
void decx::fft::IFFT1D_CR_b2_f(decx::_Vector<de::CPf>* src, decx::_Vector<float>* dst, de::DH* handle, decx::fft::FFT_Configs* config,
    cudaStream_t& S)
{
    const size_t src_len = src->length;
    const int _fragment = src_len / 2;

    // allocate device memory and set some pointers
    decx::PtrInfo<de::CPf> dev_tmp;
    decx::alloc::MIF<float2> dev_tmp1, dev_tmp2;
    if (decx::alloc::_device_malloc<de::CPf>(&dev_tmp, 2 * src_len * sizeof(de::CPf))) {
        decx::err::AllocateFailure(handle);
        return;
    }
    
    dev_tmp1.mem = reinterpret_cast<float2*>(dev_tmp.ptr);
    dev_tmp2.mem = reinterpret_cast<float2*>(dev_tmp1.mem + src_len);
    
    // copy the data from host to device
    checkCudaErrors(cudaMemcpyAsync(dev_tmp1.mem, src->Vec.ptr, src_len * sizeof(de::CPf), cudaMemcpyHostToDevice, S));

    decx::fft::IFFT1D_2N_HFkernel_caller_C2R(_fragment, &dev_tmp1, &dev_tmp2, config, S);

    // copy the data from device back to host
    if (dev_tmp1.leading) {
        checkCudaErrors(cudaMemcpyAsync(
            dst->Vec.ptr, dev_tmp1.mem, src_len * sizeof(float), cudaMemcpyDeviceToHost, S));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(
            dst->Vec.ptr, dev_tmp2.mem, src_len * sizeof(float), cudaMemcpyDeviceToHost, S));
    }

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_dealloc_D(dev_tmp.block);
    
    decx::Success(handle);
}




static
void decx::fft::IFFT1D_CC_b2_f(decx::_Vector<de::CPf>* src, decx::_Vector<de::CPf>* dst, de::DH* handle, decx::fft::FFT_Configs* config,
    cudaStream_t& S)
{
    const size_t src_len = src->length;
    const int _fragment = src_len / 2;

    // allocate device memory and set some pointers
    decx::PtrInfo<de::CPf> dev_tmp;
    decx::alloc::MIF<float2> dev_tmp1, dev_tmp2;
    if (decx::alloc::_device_malloc<de::CPf>(&dev_tmp, 2 * src_len * sizeof(de::CPf))) {
        decx::err::AllocateFailure(handle);
        return;
    }

    dev_tmp1.mem = reinterpret_cast<float2*>(dev_tmp.ptr);
    dev_tmp2.mem = reinterpret_cast<float2*>(dev_tmp1.mem + src_len);

    // copy the data from host to device
    checkCudaErrors(cudaMemcpyAsync(dev_tmp1.mem, src->Vec.ptr, src_len * sizeof(de::CPf), cudaMemcpyHostToDevice, S));

    decx::fft::IFFT1D_2N_HFkernel_caller_C2C(_fragment, &dev_tmp1, &dev_tmp2, config, S);

    // copy the data from device back to host
    if (dev_tmp1.leading) {
        checkCudaErrors(cudaMemcpyAsync(
            dst->Vec.ptr, dev_tmp1.mem, src_len * sizeof(de::CPf), cudaMemcpyDeviceToHost, S));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(
            dst->Vec.ptr, dev_tmp2.mem, src_len * sizeof(de::CPf), cudaMemcpyDeviceToHost, S));
    }

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_dealloc_D(dev_tmp.block);
    
    decx::Success(handle);
}




static 
void decx::fft::IFFT1D_2N_HFkernel_caller_C2R(const size_t                    _fragment,
                                              decx::alloc::MIF<float2>*        dev_tmp1,
                                              decx::alloc::MIF<float2>*        dev_tmp2,
                                              decx::fft::FFT_Configs*        config,
                                              cudaStream_t&                    S)
{
    __cu_MixRadix_sort_C_halved << <_cu_ceil(_fragment, 512), 512, 0, S >> > (
        reinterpret_cast<float4*>(dev_tmp1->mem), reinterpret_cast<float4*>(dev_tmp2->mem), _fragment, config->_base_num);

    int consec_frag = 1 << (2 * config->_consecutive_4);
    cu_IFFT1D_b4_CC_consec128_halved << <_cu_ceil(_fragment, consec_frag), 256, 0, S >> > (
        reinterpret_cast<float4*>(dev_tmp2->mem),
        reinterpret_cast<float4*>(dev_tmp1->mem),
        consec_frag / 4,
        config->_consecutive_4);

    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
    
    //int warp_proc_domain = 1, warp_len = 1;
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
                cu_IFFT1D_b2_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_IFFT1D_b2_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        case 4:
            if (dev_tmp1->leading) {
                cu_IFFT1D_b4_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_IFFT1D_b4_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        default:
            break;
        }

        warp_len *= current_base;
    }

    if (dev_tmp1->leading) {
        cu_IFFT1D_b2_single_clamp_last_CR << < decx::utils::ceil<size_t>(_fragment, 512), 512, 0, S >> > (
            dev_tmp1->mem, reinterpret_cast<float*>(dev_tmp2->mem), _fragment * 2, _fragment, _fragment);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
    }
    else {
        cu_IFFT1D_b2_single_clamp_last_CR << < decx::utils::ceil<size_t>(_fragment, 512), 512, 0, S >> > (
            dev_tmp2->mem, reinterpret_cast<float*>(dev_tmp1->mem), _fragment * 2, _fragment, _fragment);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
    }
}




static 
void decx::fft::IFFT1D_2N_HFkernel_caller_C2C(const size_t                    _fragment,
                                              decx::alloc::MIF<float2>*        dev_tmp1,
                                              decx::alloc::MIF<float2>*        dev_tmp2,
                                              decx::fft::FFT_Configs*        config,
                                              cudaStream_t&                    S)
{
    __cu_MixRadix_sort_C_halved << <_cu_ceil(_fragment, 512), 512, 0, S >> > (
        reinterpret_cast<float4*>(dev_tmp1->mem), reinterpret_cast<float4*>(dev_tmp2->mem), _fragment, config->_base_num);

    int consec_frag = 1 << (2 * config->_consecutive_4);
    cu_IFFT1D_b4_CC_consec128_halved << <decx::utils::ceil<size_t>(_fragment, consec_frag), 256, 0, S >> > (
        reinterpret_cast<float4*>(dev_tmp2->mem),
        reinterpret_cast<float4*>(dev_tmp1->mem),
        consec_frag / 4,
        config->_consecutive_4);

    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
    
    //int warp_proc_domain = 1, warp_len = 1;
    size_t warp_proc_domain = consec_frag, warp_len = consec_frag;
    size_t total_thr_num;
    for (int i = config->_consecutive_4; i < config->_base_num; ++i)
    {
        int current_base = config->_base.operator[](i);
        warp_proc_domain *= current_base;
        total_thr_num = _fragment / current_base;

        switch (current_base)
        {
        case 2:
            if (dev_tmp1->leading) {
                cu_IFFT1D_b2_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_IFFT1D_b2_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        case 4:
            if (dev_tmp1->leading) {
                cu_IFFT1D_b4_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_IFFT1D_b4_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        default:
            break;
        }

        warp_len *= current_base;
    }

    if (dev_tmp1->leading) {
        cu_IFFT1D_b2_single_clamp_last_CC << < decx::utils::ceil<size_t>(_fragment, 512), 512, 0, S >> > (
            dev_tmp1->mem, dev_tmp2->mem, _fragment * 2, _fragment, _fragment);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
    }
    else {
        cu_IFFT1D_b2_single_clamp_last_CC << < decx::utils::ceil<size_t>(_fragment, 512), 512, 0, S >> > (
            dev_tmp2->mem, dev_tmp1->mem, _fragment * 2, _fragment, _fragment);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
    }
}




static
void decx::fft::IFFT1D_C2R_MixBase_f(decx::_Vector<de::CPf>* src, decx::_Vector<float>* dst, de::DH* handle, decx::fft::FFT_Configs* config,
    cudaStream_t& S)
{
    const size_t src_len = src->length;
    
    // allocate device memory and set some pointers
    decx::PtrInfo<de::CPf> dev_tmp;

    decx::alloc::MIF<float2> dev_tmp1, dev_tmp2;
    const size_t req_size = decx::utils::ceil<size_t>(src_len, 8) * 8;
    if (decx::alloc::_device_malloc<de::CPf>(&dev_tmp, 2 * src_len * sizeof(de::CPf))) {
        decx::err::AllocateFailure(handle);
        return;
    }
    dev_tmp1.mem = reinterpret_cast<float2*>(dev_tmp.ptr);
    dev_tmp2.mem = reinterpret_cast<float2*>(dev_tmp1.mem + src_len);

    checkCudaErrors(cudaMemcpyAsync(
        dev_tmp1.mem, src->Vec.ptr, src->total_bytes, cudaMemcpyHostToDevice, S));

    if (config->_can_halve) {
        const size_t _fragment = src_len / 2;

        decx::fft::IFFT1D_C2R_MixRadix_f_kernel_caller_halved(&dev_tmp1, &dev_tmp2, config, _fragment, S);
    }
    else {
        decx::fft::IFFT1D_C2R_MixRadix_f_kernel_caller(&dev_tmp1, &dev_tmp2, config, src_len, S);
    }

    if (dev_tmp1.leading) {
        checkCudaErrors(cudaMemcpyAsync(
            dst->Vec.ptr, dev_tmp1.mem, dst->total_bytes, cudaMemcpyDeviceToHost, S));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(
            dst->Vec.ptr, dev_tmp2.mem, dst->total_bytes, cudaMemcpyDeviceToHost, S));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_dealloc_D(dev_tmp.block);
}




static
void decx::fft::IFFT1D_C2C_MixBase_f(decx::_Vector<de::CPf>* src, decx::_Vector<de::CPf>* dst, de::DH* handle, decx::fft::FFT_Configs* config,
    cudaStream_t& S)
{
    const size_t src_len = src->length;
    
    // allocate device memory and set some pointers
    decx::PtrInfo<de::CPf> dev_tmp;

    decx::alloc::MIF<float2> dev_tmp1, dev_tmp2;
    const size_t req_size = _cu_ceil(src_len, 8) * 8;
    if (decx::alloc::_device_malloc<de::CPf>(&dev_tmp, 2 * src_len * sizeof(de::CPf))) {
        decx::err::AllocateFailure(handle);
        return;
    }
    dev_tmp1.mem = reinterpret_cast<float2*>(dev_tmp.ptr);
    dev_tmp2.mem = reinterpret_cast<float2*>(dev_tmp1.mem + src_len);

    checkCudaErrors(cudaMemcpyAsync(
        dev_tmp1.mem, src->Vec.ptr, src->total_bytes, cudaMemcpyHostToDevice, S));

    if (config->_can_halve) {
        const int _fragment = src_len / 2;

        decx::fft::IFFT1D_C2C_MixRadix_f_kernel_caller_halved(&dev_tmp1, &dev_tmp2, config, _fragment, S);
    }
    else {
        decx::fft::IFFT1D_C2C_MixRadix_f_kernel_caller(&dev_tmp1, &dev_tmp2, config, src_len, S);
    }

    if (dev_tmp1.leading) {
        checkCudaErrors(cudaMemcpyAsync(
            dst->Vec.ptr, dev_tmp1.mem, dst->total_bytes, cudaMemcpyDeviceToHost, S));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(
            dst->Vec.ptr, dev_tmp2.mem, dst->total_bytes, cudaMemcpyDeviceToHost, S));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_dealloc_D(dev_tmp.block);
}




static 
void decx::fft::IFFT1D_C2R_MixRadix_f_kernel_caller_halved(decx::alloc::MIF<float2>* dev_tmp1, decx::alloc::MIF<float2>* dev_tmp2,
    decx::fft::FFT_Configs* config, size_t _fragment, cudaStream_t &S)
{
    __cu_MixRadix_sort_C_halved << <_cu_ceil(_fragment, 512), 512, 0, S >> > (
        reinterpret_cast<float4*>(dev_tmp1->mem), reinterpret_cast<float4*>(dev_tmp2->mem), _fragment, config->_base_num);
    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);

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
                cu_IFFT1D_b2_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_IFFT1D_b2_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        case 3:
            if (dev_tmp1->leading) {
                cu_IFFT1D_b3_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_IFFT1D_b3_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        case 4:
            if (dev_tmp1->leading) {
                cu_IFFT1D_b4_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_IFFT1D_b4_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        case 5:
            if (dev_tmp1->leading) {
                cu_IFFT1D_b5_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_IFFT1D_b5_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;
        default:
            break;
        }

        warp_len *= current_base;
    }

    if (dev_tmp1->leading) {
        cu_IFFT1D_b2_single_clamp_last_CR << < _cu_ceil(_fragment, 512), 512, 0, S >> > (
            dev_tmp1->mem, reinterpret_cast<float*>(dev_tmp2->mem), _fragment * 2, _fragment, _fragment);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
    }
    else {
        cu_IFFT1D_b2_single_clamp_last_CR << < _cu_ceil(_fragment, 512), 512, 0, S >> > (
            dev_tmp2->mem, reinterpret_cast<float*>(dev_tmp1->mem), _fragment * 2, _fragment, _fragment);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
    }
}




static
void decx::fft::IFFT1D_C2C_MixRadix_f_kernel_caller_halved(decx::alloc::MIF<float2>* dev_tmp1, decx::alloc::MIF<float2>* dev_tmp2,
    decx::fft::FFT_Configs* config, size_t _fragment, cudaStream_t& S)
{
    __cu_MixRadix_sort_C_halved << <_cu_ceil(_fragment, 512), 512, 0, S >> > (
        reinterpret_cast<float4*>(dev_tmp1->mem), reinterpret_cast<float4*>(dev_tmp2->mem), _fragment, config->_base_num);
    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);

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
                cu_IFFT1D_b2_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_IFFT1D_b2_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        case 3:
            if (dev_tmp1->leading) {
                cu_IFFT1D_b3_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_IFFT1D_b3_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        case 4:
            if (dev_tmp1->leading) {
                cu_IFFT1D_b4_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_IFFT1D_b4_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;

        case 5:
            if (dev_tmp1->leading) {
                cu_IFFT1D_b5_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
            }
            else {
                cu_IFFT1D_b5_CC_halved << < _cu_ceil(total_thr_num, 512), 512, 0, S >> > (
                    reinterpret_cast<float4*>(dev_tmp2->mem),
                    reinterpret_cast<float4*>(dev_tmp1->mem),
                    total_thr_num, warp_proc_domain, warp_len);

                decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
            }
            break;
        default:
            break;
        }

        warp_len *= current_base;
    }

    if (dev_tmp1->leading) {
        cu_IFFT1D_b2_single_clamp_last_CC << < _cu_ceil(_fragment, 512), 512, 0, S >> > (
            dev_tmp1->mem, dev_tmp2->mem, _fragment * 2, _fragment, _fragment);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);
    }
    else {
        cu_IFFT1D_b2_single_clamp_last_CC << < _cu_ceil(_fragment, 512), 512, 0, S >> > (
            dev_tmp2->mem, dev_tmp1->mem, _fragment * 2, _fragment, _fragment);

        decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp1, dev_tmp2);
    }
}




static
void decx::fft::IFFT1D_C2R_MixRadix_f_kernel_caller(decx::alloc::MIF<float2>* dev_tmp1, decx::alloc::MIF<float2>* dev_tmp2,
    decx::fft::FFT_Configs* config, size_t _fragment, cudaStream_t& S)
{
    __cu_MixRadix_sort_C << <_cu_ceil(_fragment, 512), 512, 0, S >> > (
        dev_tmp1->mem, dev_tmp2->mem, _fragment, config->_base_num);
    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);

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
            IFFT_KERNEL_FRAME_C2R(cu_IFFT1D_b3_CC, cu_IFFT1D_b3_CR_last);
            break;

        case 5:
            IFFT_KERNEL_FRAME_C2R(cu_IFFT1D_b5_CC, cu_IFFT1D_b5_CR_last);
            break;
        default:
            break;
        }

        warp_len *= current_base;
    }
}




static
void decx::fft::IFFT1D_C2C_MixRadix_f_kernel_caller(decx::alloc::MIF<float2>* dev_tmp1, decx::alloc::MIF<float2>* dev_tmp2,
    decx::fft::FFT_Configs* config, size_t _fragment, cudaStream_t& S)
{
    __cu_MixRadix_sort_C << <_cu_ceil(_fragment, 512), 512, 0, S >> > (
        dev_tmp1->mem, dev_tmp2->mem, _fragment, config->_base_num);
    decx::utils::set_mutex_memory_state<float2, float2>(dev_tmp2, dev_tmp1);

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
            IFFT_KERNEL_FRAME_C2C(cu_IFFT1D_b3_CC, cu_IFFT1D_b3_CC_last);
            break;

        case 5:
            IFFT_KERNEL_FRAME_C2C(cu_IFFT1D_b5_CC, cu_IFFT1D_b5_CC_last);
            break;
        default:
            break;
        }

        warp_len *= current_base;
    }
}


#endif      // #ifndef _IFFT1D_SUB_FUNCS_H_