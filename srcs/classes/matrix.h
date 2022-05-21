/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once


#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "../core/basic.h"
#include "classes_util.h"
#include "../core/allocators.h"
#include "../core/thread_management/thread_pool.h"
#include "store_types.h"



#define _MATRIX_ALIGN_4B_ 8
#define _MATRIX_ALIGN_2B_ 16
#define _MATRIX_ALIGN_8B_ 4
#define _MATRIX_ALIGN_1B_ 32


/**
* in host, allocate page-locaked memory in 8-times both on width and height
* ensure the utilization of __m128 and __m256, as well as multi threads
*/
namespace de {
    template<typename T>
    class _DECX_API_ Matrix
    {
    public:
        Matrix() {}


        virtual uint Width() = 0;


        virtual uint Height() = 0;


        virtual size_t TotalBytes() = 0;


        /* return the reference of the element in the matrix, which locates on specific row and colume
        * \params row -> where the element locates on row
        * \params col -> where the element locates on colume
        */
        virtual T& index(const int row, const int col) = 0;

        
        virtual void release() = 0;


        virtual de::Matrix<T>& operator=(de::Matrix<T>& src) = 0;


        ~Matrix() {}
    };
}


#ifndef _DECX_COMBINED_


/**
* The data storage structure is shown below
*            <-------------- width ------------->
*            <---------------- pitch ------------------>
*             <-dpitch->
*            [[x x x x x x... ...    ... ...x x x....] T        
*            [[x x x x x x... ...    ... ...x x x....] |    
*            [[x x x x x x... ...    ... ...x x x....] |    
*            ...                                     ...  |    height    
*            ...                                     ...  |        
*            ...                                     ...  |        
*            [[x x x x x x... ...    ... ...x x x....] _
*/


namespace decx
{
    template<typename T>
    class _Matrix : public de::Matrix<T>
    {
    private:
        // call AFTER attributes are assigned !
        void re_alloc_data_space();

        // call AFTER attributes are assigned !
        void alloc_data_space();


        void _attribute_assign(const uint _width, const uint _height, const int store_type);

    public:
        uint width, height;

        unsigned short Store_Type;

        size_t element_num, total_bytes;
        uint pitch;            // the true width (NOT IN BYTES), it is aligned with 8

        decx::PtrInfo<T> Mat;


        size_t _element_num,    // true_width * true_height
            _total_bytes;    // true_width * true_height * sizeof(T)


        void construct(uint width, uint height, const int flag);


        void re_construct(uint width, uint height, const int flag);


        _Matrix();


        _Matrix(const uint _width, const uint _height, const int store_type);


        virtual uint Width() { return this->width; }


        virtual uint Height() { return this->height; }


        virtual size_t TotalBytes() { return this->total_bytes; }


        virtual T& index(const int row, const int col);


        virtual void release();


        virtual de::Matrix<T>& operator=(de::Matrix<T> &src);


        virtual ~_Matrix();
    };
}


namespace de
{
    template <typename T>
    de::Matrix<T>* CreateMatrixPtr();


    template <typename T>
    de::Matrix<T>& CreateMatrixRef();


    template <typename T>
    de::Matrix<T>* CreateMatrixPtr(const uint _width, const uint _height, const int store_type);


    template <typename T>
    de::Matrix<T>& CreateMatrixRef(const uint _width, const uint _height, const int store_type);
}



template <typename T>
void decx::_Matrix<T>::alloc_data_space()
{
    switch (this->Store_Type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        if (decx::alloc::_host_virtual_page_malloc<T>(&this->Mat, this->_total_bytes)) {
            SetConsoleColor(4);
            printf("Matrix malloc failed! Please check if there is enough space in your device.");
            ResetConsoleColor;
            return;
        }
        break;

#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_malloc<T>(&this->Mat, this->_total_bytes)) {
            SetConsoleColor(4);
            printf("Matrix malloc failed! Please check if there is enough space in your device.");
            ResetConsoleColor;
            return;
        }
        break;
#endif
    }
}


template <typename T>
void decx::_Matrix<T>::re_alloc_data_space()
{
    switch (this->Store_Type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        if (decx::alloc::_host_virtual_page_realloc<T>(&this->Mat, this->_total_bytes)) {
            SetConsoleColor(4);
            printf("Matrix malloc failed! Please check if there is enough space in your device.");
            ResetConsoleColor;
            return;
        }
        break;

#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_realloc<T>(&this->Mat, this->_total_bytes)) {
            SetConsoleColor(4);
            printf("Matrix malloc failed! Please check if there is enough space in your device.");
            ResetConsoleColor;
            return;
        }
        break;
#endif
    }
}



template <typename T>
void decx::_Matrix<T>::construct(uint _width, uint _height, const int flag)
{
    this->_attribute_assign(_width, _height, flag);

    this->alloc_data_space();
}


template <typename T>
void decx::_Matrix<T>::re_construct(uint _width, uint _height, const int flag)
{
    // If all the parameters are the same, it is meaningless to re-construt the data
    if (this->width != _width || this->height != _height || this->Store_Type != flag)
    {
        this->_attribute_assign(_width, _height, flag);

        this->re_alloc_data_space();
    }
}




void decx::_Matrix<float>::_attribute_assign(const uint _width, const uint _height, const int store_type)
{
    this->width = _width;
    this->height = _height;

    this->Store_Type = store_type;

    this->pitch = decx::utils::ceil<int>(_width, _MATRIX_ALIGN_4B_) * _MATRIX_ALIGN_4B_;

    this->element_num = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->total_bytes = (this->element_num) * sizeof(float);

    this->_element_num = static_cast<size_t>(this->pitch) * static_cast<size_t>(_height);
    this->_total_bytes = (this->_element_num) * sizeof(float);
}



void decx::_Matrix<int>::_attribute_assign(const uint _width, const uint _height, const int store_type)
{
    this->width = _width;
    this->height = _height;

    this->Store_Type = store_type;

    this->pitch = decx::utils::ceil<int>(_width, _MATRIX_ALIGN_4B_) * _MATRIX_ALIGN_4B_;

    this->element_num = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->total_bytes = (this->element_num) * sizeof(int);

    this->_element_num = static_cast<size_t>(this->pitch) * static_cast<size_t>(_height);
    this->_total_bytes = (this->_element_num) * sizeof(int);
}



void decx::_Matrix<double>::_attribute_assign(const uint _width, const uint _height, const int store_type)
{
    this->width = _width;
    this->height = _height;

    this->Store_Type = store_type;

    this->pitch = decx::utils::ceil<int>(_width, _MATRIX_ALIGN_8B_) * _MATRIX_ALIGN_8B_;

    this->element_num = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->total_bytes = (this->element_num) * sizeof(double);

    this->_element_num = static_cast<size_t>(this->pitch) * static_cast<size_t>(_height);
    this->_total_bytes = (this->_element_num) * sizeof(double);
}



#ifndef GNU_CPUcodes
void decx::_Matrix<de::Half>::_attribute_assign(const uint _width, const uint _height, const int store_type)
{
    this->width = _width;
    this->height = _height;

    this->Store_Type = store_type;

    this->pitch = decx::utils::ceil<int>(_width, _MATRIX_ALIGN_2B_) * _MATRIX_ALIGN_2B_;

    this->element_num = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->total_bytes = (this->element_num) * sizeof(de::Half);

    this->_element_num = static_cast<size_t>(this->pitch) * static_cast<size_t>(_height);
    this->_total_bytes = (this->_element_num) * sizeof(de::Half);
}
#endif


void decx::_Matrix<uchar>::_attribute_assign(const uint _width, const uint _height, const int store_type)
{
    this->width = _width;
    this->height = _height;

    this->Store_Type = store_type;

    this->pitch = decx::utils::ceil<int>(_width, _MATRIX_ALIGN_1B_) * _MATRIX_ALIGN_1B_;

    this->element_num = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->total_bytes = (this->element_num) * sizeof(uchar);

    this->_element_num = static_cast<size_t>(this->pitch) * static_cast<size_t>(_height);
    this->_total_bytes = (this->_element_num) * sizeof(uchar);
}



void decx::_Matrix<de::CPf>::_attribute_assign(const uint _width, const uint _height, const int store_type)
{
    this->width = _width;
    this->height = _height;

    this->Store_Type = store_type;

    this->pitch = decx::utils::ceil<int>(_width, _MATRIX_ALIGN_8B_) * _MATRIX_ALIGN_8B_;

    this->element_num = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->total_bytes = (this->element_num) * sizeof(de::CPf);

    this->_element_num = static_cast<size_t>(this->pitch) * static_cast<size_t>(_height);
    this->_total_bytes = (this->_element_num) * sizeof(de::CPf);
}



template <typename T>
void decx::_Matrix<T>::release()
{
    switch (this->Store_Type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_dealloc(&this->Mat);
        break;

#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_dealloc(&this->Mat);
        break;
#endif

    default:
        break;
    }
}


template<typename T>
decx::_Matrix<T>::_Matrix()
{
    this->_attribute_assign(0, 0, 0);
}


template<typename T>
decx::_Matrix<T>::_Matrix(const uint _width, const uint _height, const int store_type)
{
    this->construct(_width, _height, store_type);
}




template<typename T>
T& decx::_Matrix<T>::index(const int row, const int col)
{
    return (this->Mat.ptr)[(this->pitch) * row + col];
}



template <typename T>
decx::_Matrix<T>::~_Matrix()
{
    if (this->Mat.ptr != NULL) {
        this->release();
    }
}



// Matrix and Tensor creation


template <typename T>
de::Matrix<T>& de::CreateMatrixRef()
{
    return *(new decx::_Matrix<T>());
}


template <typename T>
de::Matrix<T>* de::CreateMatrixPtr()
{
    return new decx::_Matrix<T>();
}

template _DECX_API_ _INT_& de::CreateMatrixRef();

template _DECX_API_ _FLOAT_& de::CreateMatrixRef();

template _DECX_API_ _DOUBLE_& de::CreateMatrixRef();

template _DECX_API_ _UCHAR_& de::CreateMatrixRef();

template _DECX_API_ _CPF_& de::CreateMatrixRef();

#ifndef GNU_CPUcodes
template _DECX_API_ _HALF_& de::CreateMatrixRef();
#endif


template _DECX_API_ _INT_* de::CreateMatrixPtr();

template _DECX_API_ _FLOAT_* de::CreateMatrixPtr();

template _DECX_API_ _DOUBLE_* de::CreateMatrixPtr();

template _DECX_API_ _UCHAR_* de::CreateMatrixPtr();

template _DECX_API_ _CPF_* de::CreateMatrixPtr();

#ifndef GNU_CPUcodes
template _DECX_API_ _HALF_* de::CreateMatrixPtr();
#endif



template <typename T>
de::Matrix<T>& de::CreateMatrixRef(const uint _width, const uint _height, const int store_type)
{
    return *(new decx::_Matrix<T>(_width, _height, store_type));
}



template <typename T>
de::Matrix<T>* de::CreateMatrixPtr(const uint _width, const uint _height, const int store_type)
{
    return new decx::_Matrix<T>(_width, _height, store_type);
}

template _DECX_API_ _INT_& de::CreateMatrixRef(const uint _width, const uint _height, const int store_type);

template _DECX_API_ _FLOAT_& de::CreateMatrixRef(const uint _width, const uint _height, const int store_type);

template _DECX_API_ _DOUBLE_& de::CreateMatrixRef(const uint _width, const uint _height, const int store_type);

template _DECX_API_ _UCHAR_& de::CreateMatrixRef(const uint _width, const uint _height, const int store_type);

template _DECX_API_ _CPF_& de::CreateMatrixRef(const uint _width, const uint _height, const int store_type);

#ifndef GNU_CPUcodes
template _DECX_API_ _HALF_& de::CreateMatrixRef(const uint _width, const uint _height, const int store_type);
#endif


template _DECX_API_ _INT_* de::CreateMatrixPtr(const uint _width, const uint _height, const int store_type);

template _DECX_API_ _FLOAT_* de::CreateMatrixPtr(const uint _width, const uint _height, const int store_type);

template _DECX_API_ _DOUBLE_* de::CreateMatrixPtr(const uint _width, const uint _height, const int store_type);

template _DECX_API_ _UCHAR_* de::CreateMatrixPtr(const uint _width, const uint _height, const int store_type);

template _DECX_API_ _CPF_* de::CreateMatrixPtr(const uint _width, const uint _height, const int store_type);

#ifndef GNU_CPUcodes
template _DECX_API_ _HALF_* de::CreateMatrixPtr(const uint _width, const uint _height, const int store_type);
#endif



template <typename T>
de::Matrix<T>& decx::_Matrix<T>::operator=(de::Matrix<T>& src)
{
    const decx::_Matrix<T>& ref_src = dynamic_cast<decx::_Matrix<T>&>(src);

    this->Mat.block = ref_src.Mat.block;

    this->_attribute_assign(ref_src.width, ref_src.height, ref_src.Store_Type);

    switch (ref_src.Store_Type)
    {
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_malloc_same_place(&this->Mat);
        break;

    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_malloc_same_place(&this->Mat);
        break;
    }

    return *this;
}



#endif    //#ifndef _DECX_COMBINED_

#endif