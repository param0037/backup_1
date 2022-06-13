/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _VECTOR_H_
#define _VECTOR_H_


#include "../core/basic.h"
#include "../core/allocators.h"
#include "../handles/decx_handles.h"
#include "classes_util.h"
#include "store_types.h"

// 39193 2022.4.28 12:49

#define _VECTOR_ALIGN_4B_ 8        // fp32, int32
#define _VECTOR_ALIGN_8B_ 4        // fp64
#define _VECTOR_ALIGN_2B_ 16       // fp16
#define _VECTOR_ALIGN_1B_ 32       // uchar


namespace de
{
    template<typename T>
    class _DECX_API_ Vector
    {
    public:
        Vector() {}


        virtual uint Len() = 0;

        virtual T &index(size_t index) = 0;


        virtual void release() = 0;


        virtual de::Vector<T>& operator=(de::Vector<T>& src) = 0;


        virtual ~Vector() {}
    };
}

/*
* Data storage structure
* 
* <---------------- _length -------------------->
* <------------- length ----------->
* [x x x x x x x x x x x x x x x x x 0 0 0 0 0 0]
*/

#ifndef _DECX_COMBINED_


namespace decx
{
    template <typename T>
    class _Vector : public de::Vector<T>
    {
    private:
        void _attribute_assign(size_t len, const int flag);


        void alloc_data_space();


        void re_alloc_data_space();

    public:
        int _store_type;
        size_t length,
            _length,    // It is aligned with 8
            total_bytes;

        decx::PtrInfo<T> Vec;


        void construct(size_t length, const int flag);


        void re_construct(size_t length, const int flag);


        _Vector();


        _Vector(size_t length, const int flag);


        virtual uint Len() { return this->length; }


        virtual T& index(size_t index) {
            return this->Vec.ptr[index];
        }


        virtual void release();


        virtual de::Vector<T> &operator=(de::Vector<T> &src);


        virtual ~_Vector() {}
    };
}



//template <typename T>
//void decx::_Vector<T>::_attribute_assign(size_t len, const int flag)
//{
//    this->length = length;
//    switch (sizeof(T)) 
//    {
//    case 4:
//        this->_length = decx::utils::ceil<size_t>(length, (size_t)_VECTOR_ALIGN_4B_) * (size_t)_VECTOR_ALIGN_4B_;
//        break;
//
//    case 8:
//        this->_length = decx::utils::ceil<size_t>(length, (size_t)_VECTOR_ALIGN_8B_) * (size_t)_VECTOR_ALIGN_8B_;
//        break;
//
//    case 2:
//        this->_length = decx::utils::ceil<size_t>(length, (size_t)_VECTOR_ALIGN_2B_) * (size_t)_VECTOR_ALIGN_2B_;
//        break;
//    }
//    
//    this->total_bytes = this->_length * sizeof(T);
//}



void decx::_Vector<float>::_attribute_assign(size_t len, const int flag)
{
    this->length = len;

    this->_length = decx::utils::ceil<size_t>(len, _VECTOR_ALIGN_4B_) * _VECTOR_ALIGN_4B_;

    this->total_bytes = this->_length * sizeof(float);

    this->_store_type = flag;
}


void decx::_Vector<int>::_attribute_assign(size_t len, const int flag)
{
    this->length = len;

    this->_length = decx::utils::ceil<size_t>(len, _VECTOR_ALIGN_4B_) * _VECTOR_ALIGN_4B_;

    this->total_bytes = this->_length * sizeof(int);

    this->_store_type = flag;
}



void decx::_Vector<de::CPf>::_attribute_assign(size_t len, const int flag)
{
    this->length = len;

    this->_length = decx::utils::ceil<size_t>(len, _VECTOR_ALIGN_8B_) * _VECTOR_ALIGN_8B_;

    this->total_bytes = this->_length * sizeof(de::CPf);

    this->_store_type = flag;
}


#ifdef _DECX_CUDA_CODES_
void decx::_Vector<de::Half>::_attribute_assign(size_t len, const int flag)
{
    this->length = len;

    this->_length = decx::utils::ceil<size_t>(len, _VECTOR_ALIGN_2B_) * _VECTOR_ALIGN_2B_;

    this->total_bytes = this->_length * sizeof(de::Half);

    this->_store_type = flag;
}
#endif


void decx::_Vector<double>::_attribute_assign(size_t len, const int flag)
{
    this->length = len;

    this->_length = decx::utils::ceil<size_t>(len, _VECTOR_ALIGN_8B_) * _VECTOR_ALIGN_8B_;

    this->total_bytes = this->_length * sizeof(double);

    this->_store_type = flag;
}



void decx::_Vector<uchar>::_attribute_assign(size_t len, const int flag)
{
    this->length = len;

    this->_length = decx::utils::ceil<size_t>(len, _VECTOR_ALIGN_1B_) * _VECTOR_ALIGN_1B_;

    this->total_bytes = this->_length * sizeof(uchar);

    this->_store_type = flag;
}


template <typename T>
void decx::_Vector<T>::alloc_data_space()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        if (decx::alloc::_host_virtual_page_malloc<T>(&this->Vec, this->total_bytes)) {
            SetConsoleColor(4);
            printf("Vector malloc failed! Please check if there is enough space in your device.");
            ResetConsoleColor;
            return;
        }
        break;

#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_malloc<T>(&this->Vec, this->total_bytes)) {
            SetConsoleColor(4);
            printf("Vector malloc failed! Please check if there is enough space in your device.");
            ResetConsoleColor;
            return;
        }
        break;
#endif
    }
}



template <typename T>
void decx::_Vector<T>::re_alloc_data_space()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        if (decx::alloc::_host_virtual_page_realloc<T>(&this->Vec, this->total_bytes)) {
            SetConsoleColor(4);
            printf("Vector malloc failed! Please check if there is enough space in your device.");
            ResetConsoleColor;
            return;
        }
        break;

#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_realloc<T>(&this->Vec, this->total_bytes)) {
            SetConsoleColor(4);
            printf("Vector malloc failed! Please check if there is enough space in your device.");
            ResetConsoleColor;
            return;
        }
        break;
#endif
    }
}



template <typename T>
void decx::_Vector<T>::construct(size_t length, const int flag)
{
    this->_attribute_assign(length, flag);

    this->alloc_data_space();
}



template <typename T>
void decx::_Vector<T>::re_construct(size_t length, const int flag)
{
    if (this->length != length || this->_store_type != flag) {
        this->_attribute_assign(length, flag);

        this->re_alloc_data_space();
    }
}



template <typename T>
decx::_Vector<T>::_Vector()
{
    this->_attribute_assign(0, 0);
}



template <typename T>
decx::_Vector<T>::_Vector(size_t length, const int flag)
{
    this->_attribute_assign(length, flag);

    switch (flag)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        this->_store_type = flag;
        if (decx::alloc::_host_virtual_page_malloc(&this->Vec, this->total_bytes)) {
            SetConsoleColor(4);
            printf("Vector malloc failed! Please check if there is enough space in your device.\n");
            ResetConsoleColor;
            return;
        }
        break;

    case decx::DATA_STORE_TYPE::Page_Locked:
        this->_store_type = flag;
        if (decx::alloc::_host_fixed_page_malloc(&this->Vec, this->total_bytes)) {
            SetConsoleColor(4);
            printf("Vector malloc failed! Please check if there is enough space in your device.\n");
            ResetConsoleColor;
            return;
        }
        break;

    default:
        Print_Error_Message(4, MEANINGLESS_FLAG);
        return;
        break;
    }
}


namespace de
{
    template <typename T>
    de::Vector<T>& CreateVectorRef();


    template <typename T>
    de::Vector<T>* CreateVectorPtr();


    template <typename T>
    de::Vector<T>& CreateVectorRef(size_t len, const int flag);



    template <typename T>
    de::Vector<T>* CreateVectorPtr(size_t len, const int flag);
}



template <typename T>
de::Vector<T>& de::CreateVectorRef()
{
    return *(new decx::_Vector<T>());
}
template _DECX_API_ de::Vector<int>&            de::CreateVectorRef();
template _DECX_API_ de::Vector<float>&            de::CreateVectorRef();
#ifndef GNU_CPUcodes
template _DECX_API_ de::Vector<de::Half>&        de::CreateVectorRef();
#endif
template _DECX_API_ de::Vector<double>&            de::CreateVectorRef();
template _DECX_API_ de::Vector<de::CPf>&        de::CreateVectorRef();
template _DECX_API_ de::Vector<uchar>&        de::CreateVectorRef();



template <typename T>
de::Vector<T>* de::CreateVectorPtr()
{
    return new decx::_Vector<T>();
}
template _DECX_API_ de::Vector<int>*            de::CreateVectorPtr();
template _DECX_API_ de::Vector<float>*            de::CreateVectorPtr();
#ifndef GNU_CPUcodes
template _DECX_API_ de::Vector<de::Half>*        de::CreateVectorPtr();
#endif
template _DECX_API_ de::Vector<double>*            de::CreateVectorPtr();
template _DECX_API_ de::Vector<de::CPf>*        de::CreateVectorPtr();
template _DECX_API_ de::Vector<uchar>*        de::CreateVectorPtr();




template <typename T>
de::Vector<T>& de::CreateVectorRef(size_t len, const int flag)
{
    return *(new decx::_Vector<T>(len, flag));
}
template _DECX_API_ de::Vector<int>&            de::CreateVectorRef(size_t len, const int flag);
template _DECX_API_ de::Vector<float>&            de::CreateVectorRef(size_t len, const int flag);
#ifndef GNU_CPUcodes
template _DECX_API_ de::Vector<de::Half>&        de::CreateVectorRef(size_t len, const int flag);
#endif
template _DECX_API_ de::Vector<double>&            de::CreateVectorRef(size_t len, const int flag);
template _DECX_API_ de::Vector<de::CPf>&            de::CreateVectorRef(size_t len, const int flag);
template _DECX_API_ de::Vector<uchar>&              de::CreateVectorRef(size_t len, const int flag);



template <typename T>
de::Vector<T>* de::CreateVectorPtr(size_t len, const int flag)
{
    return new decx::_Vector<T>(len, flag);
}
template _DECX_API_ de::Vector<int>*            de::CreateVectorPtr(size_t len, const int flag);
template _DECX_API_ de::Vector<float>*            de::CreateVectorPtr(size_t len, const int flag);
#ifndef GNU_CPUcodes
template _DECX_API_ de::Vector<de::Half>*        de::CreateVectorPtr(size_t len, const int flag);
#endif
template _DECX_API_ de::Vector<double>*            de::CreateVectorPtr(size_t len, const int flag);
template _DECX_API_ de::Vector<de::CPf>*        de::CreateVectorPtr(size_t len, const int flag);
template _DECX_API_ de::Vector<uchar>*          de::CreateVectorPtr(size_t len, const int flag);



template<typename T>
void decx::_Vector<T>::release()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_dealloc<T>(&this->Vec);
        break;

    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_dealloc<T>(&this->Vec);
        break;
    }
}


template <typename T>
de::Vector<T>& decx::_Vector<T>::operator=(de::Vector<T>& src)
{
    const decx::_Vector<T>& ref_src = dynamic_cast<decx::_Vector<T>&>(src);

    this->_attribute_assign(ref_src.length, ref_src._store_type);
    
    switch (ref_src._store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_malloc_same_place(&this->Vec);
        break;

    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_malloc_same_place(&this->Vec);
        break;
    default:
        break;
    }
    
    return *this;
}


#endif    //#ifndef _DECX_COMBINED_

#endif