/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

/**
* Memory deallocators are defined in this header
*/

#pragma once



#include "../memory_management/Memory_pool.h"
#include "../../handles/decx_handles.h"


namespace decx
{
    namespace alloc
    {
        _DECX_API_ void _dealloc_Hv(decx::MemBlock* _ptr);

        template <typename _Ty>
        static void _host_virtual_page_dealloc(decx::PtrInfo<_Ty>* ptr_info);

        _DECX_API_ void _dealloc_Hf(decx::MemBlock* _ptr);


        template <typename _Ty>
        static void _host_fixed_page_dealloc(decx::PtrInfo<_Ty>* ptr_info);


        _DECX_API_ void _dealloc_D(decx::MemBlock* _ptr);


        template <typename _Ty>
        static void _device_dealloc(decx::PtrInfo<_Ty>* ptr_info);
    }
}



void decx::alloc::_dealloc_Hv(decx::MemBlock* _ptr) {
    decx::mem_pool_Hv.deallocate(_ptr);
}


void decx::alloc::_dealloc_Hf(decx::MemBlock* _ptr) {
    decx::mem_pool_Hf.deallocate(_ptr);
}


void decx::alloc::_dealloc_D(decx::MemBlock* _ptr) {
    decx::mem_pool_D.deallocate(_ptr);
}