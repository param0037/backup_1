/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _CONFIG_H_
#define _CONFIG_H_

#include "../../core/basic.h"


#ifdef _DECX_CUDA_CODES_
namespace decx
{
    typedef struct cudaProp
    {
        cudaDeviceProp prop;
        int CURRENT_DEVICE;
        bool is_init;

        cudaProp() { this->is_init = false; }
    };
}
#endif

#ifdef _DECX_CPU_CODES_
namespace decx
{
    typedef struct cpuInfo
    {
        size_t cpu_concurrency;
        bool is_init;

        cpuInfo() {
            is_init = false;
        }
    };
}
#endif


namespace decx
{
#ifdef _DECX_CUDA_CODES_
    decx::cudaProp cuP;
#endif

#ifdef _DECX_CPU_CODES_
    decx::cpuInfo cpI;
#endif
}


namespace de
{
#ifdef _DECX_CUDA_CODES_
    _DECX_API_ void InitCuda();
#endif

#ifdef _DECX_CPU_CODES_
    _DECX_API_ void InitCPUInfo();
#endif
}



#ifdef _DECX_CUDA_CODES_
_DECX_API_ void de::InitCuda()
{
    checkCudaErrors(cudaGetDevice(&decx::cuP.CURRENT_DEVICE));
    checkCudaErrors(cudaGetDeviceProperties(&decx::cuP.prop, decx::cuP.CURRENT_DEVICE));
    decx::cuP.is_init = true;
}
#endif

#ifdef _DECX_CPU_CODES_
_DECX_API_ void de::InitCPUInfo()
{
    decx::cpI.is_init = true;
    decx::cpI.cpu_concurrency = std::thread::hardware_concurrency();
}
#endif


#endif