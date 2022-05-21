/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

/**
* Memory allocators are defined in this header
*/

#ifndef _ALLOCATORS_APIS_H_
#define _ALLOCATORS_APIS_H_

#include "../core/basic.h"
#include "../core/memory_management/MemBlock.h"

#ifdef Windows
#pragma comment(lib, "../../../../bin/x64/DECX_allocation.lib")
#endif


// allocation

namespace decx
{
    namespace alloc
    {
        _DECX_API_ int _alloc_Hv(decx::MemBlock** _ptr, size_t req_size);


        template <typename _Ty>
        static int _host_virtual_page_malloc(decx::PtrInfo<_Ty>* ptr_info, size_t size);

        _DECX_API_ int _alloc_Hf(decx::MemBlock** _ptr, size_t req_size);


        template <typename _Ty>
        static int _host_fixed_page_malloc(decx::PtrInfo<_Ty>* ptr_info, size_t size);

        /**
        * @return If successed, 0; If failed -1
        */
        _DECX_API_ int _alloc_D(decx::MemBlock** _ptr, size_t req_size);


        template <typename _Ty>
        static int _device_malloc(decx::PtrInfo<_Ty>* ptr_info, size_t size);


        /** @return If successed, 0; If failed -1 */
        _DECX_API_ void _alloc_Hv_same_place(decx::MemBlock** _ptr);


        template <typename _Ty>
        static void _host_virtual_page_malloc_same_place(decx::PtrInfo<_Ty>* ptr_info);


        /** @return If successed, 0; If failed -1 */
        _DECX_API_ void _alloc_Hf_same_place(decx::MemBlock** _ptr);


        template <typename _Ty>
        static void _host_fixed_page_malloc_same_place(decx::PtrInfo<_Ty>* ptr_info);


        /** @return If successed, 0; If failed -1 */
        _DECX_API_ void _alloc_D_same_place(decx::MemBlock** _ptr);


        template <typename _Ty>
        static void _device_malloc_same_place(decx::PtrInfo<_Ty>* ptr_info);


        template <typename T>
        int _host_fixed_page_realloc(decx::PtrInfo<T>* ptr_info, size_t size);


        template <typename T>
        int _host_virtual_page_realloc(decx::PtrInfo<T>* ptr_info, size_t size);


        template <typename T>
        int _device_realloc(decx::PtrInfo<T>* ptr_info, size_t size);
    }
}

// deallocation

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



template <typename _Ty>
static void decx::alloc::_host_virtual_page_dealloc(decx::PtrInfo<_Ty>* ptr_info) {
    decx::alloc::_dealloc_Hv(ptr_info->block);
    ptr_info->ptr = NULL;
}




template <typename _Ty>
static void decx::alloc::_host_fixed_page_dealloc(decx::PtrInfo<_Ty>* ptr_info) {
    decx::alloc::_dealloc_Hf(ptr_info->block);
    ptr_info->ptr = NULL;
}



template <typename _Ty>
static void decx::alloc::_device_dealloc(decx::PtrInfo<_Ty>* ptr_info) {
    decx::alloc::_dealloc_D(ptr_info->block);
    ptr_info->ptr = NULL;
}




template <typename T>
static int decx::alloc::_host_virtual_page_malloc(decx::PtrInfo<T>* ptr_info, size_t size)
{
    int ans = decx::alloc::_alloc_Hv(&ptr_info->block, size);
    ptr_info->_sync_type();
    return ans;
}



template <typename T>
static void decx::alloc::_host_virtual_page_malloc_same_place(decx::PtrInfo<T>* ptr_info)
{
    decx::alloc::_alloc_Hv_same_place(&ptr_info->block);
    ptr_info->_sync_type();
}



template <typename T>
static int decx::alloc::_host_fixed_page_malloc(decx::PtrInfo<T>* ptr_info, size_t size)
{
    int ans = decx::alloc::_alloc_Hf(&ptr_info->block, size);
    ptr_info->_sync_type();
    return ans;
}



template <typename T>
static void decx::alloc::_host_fixed_page_malloc_same_place(decx::PtrInfo<T>* ptr_info)
{
    decx::alloc::_alloc_Hf_same_place(&ptr_info->block);
    ptr_info->_sync_type();
}



template <typename T>
static int decx::alloc::_device_malloc(decx::PtrInfo<T>* ptr_info, size_t size)
{
    int ans = decx::alloc::_alloc_D(&ptr_info->block, size);
    ptr_info->_sync_type();
    return ans;
}


template <typename T>
static void decx::alloc::_device_malloc_same_place(decx::PtrInfo<T>* ptr_info)
{
    decx::alloc::_alloc_D_same_place(&ptr_info->block);
    ptr_info->_sync_type();
}



template <typename T>
int decx::alloc::_host_fixed_page_realloc(decx::PtrInfo<T>* ptr_info, size_t size)
{
    if (ptr_info->block != NULL) {
        if (ptr_info->block->_ptr != NULL) {            // if it is previously allocated
            decx::alloc::_dealloc_Hf(ptr_info->block);
        }
    }
    // reallocate new memory of new size
    int ans = decx::alloc::_alloc_Hf(&ptr_info->block, size);
    ptr_info->_sync_type();

    return ans;
}


template <typename T>
int decx::alloc::_host_virtual_page_realloc(decx::PtrInfo<T>* ptr_info, size_t size)
{
    if (ptr_info->block != NULL) {
        if (ptr_info->block->_ptr != NULL) {            // if it is previously allocated
            decx::alloc::_dealloc_Hv(ptr_info->block);
        }
    }
    // reallocate new memory of new size
    int ans = decx::alloc::_alloc_Hv(&ptr_info->block, size);
    ptr_info->_sync_type();

    return ans;
}



template <typename T>
int decx::alloc::_device_realloc(decx::PtrInfo<T>* ptr_info, size_t size)
{
    if (ptr_info->block != NULL) {
        if (ptr_info->block->_ptr != NULL) {            // if it is previously allocated
            decx::alloc::_dealloc_D(ptr_info->block);
        }
    }
    // reallocate new memory of new size
    int ans = decx::alloc::_alloc_D(&ptr_info->block, size);
    ptr_info->_sync_type();

    return ans;
}



#endif