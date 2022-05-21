/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _MATRIXARRAY_H_
#define _MATRIXARRAY_H_


#include "../core/basic.h"
#include "matrix.h"
#include "../core/allocators.h"
#include "store_types.h"


namespace de
{
    /*
    * This class is for matrices array, the matrices included share the same sizes, and store
    * one by one in the memory block, without gap; Compared with de::Tensor<T>, the channel "z"
    * is separated.
    */
    template <typename T>
    class _DECX_API_ MatrixArray
    {
    public:
        uint ArrayNumber;        


        virtual uint Width() = 0;


        virtual uint Height() = 0;


        virtual uint MatrixNumber() = 0;


        virtual T& index(uint row, uint col, size_t _seq) = 0;


        virtual de::MatrixArray<T>& operator=(de::MatrixArray<T>& src) = 0;


        virtual void release() = 0;
    };
}


#ifndef _DECX_COMBINED_

namespace decx
{
    template <typename T>
    class _MatrixArray : public de::MatrixArray<T>
    {
    private:
        // call AFTER attributes are assigned !
        void re_alloc_data_space();

        // call AFTER attributes are assigned !
        void alloc_data_space();


        void _attribute_assign(uint width, uint height, uint MatrixNum, const int flag);

    public:
        int _store_type;

        decx::PtrInfo<T> MatArr;
        decx::PtrInfo<T*> MatptrArr;

        uint width, height;

        size_t element_num, _element_num,
            total_bytes,    // The real total bytes of the MatrixArray memory block, ATTENTION : NO '_' at the front
            ArrayNumber;    // The number of matrices that share the same sizes

        size_t plane, _plane;

        uint pitch,            // the true width (NOT IN BYTES)
            _height;        // the true height

        void construct(uint width, uint height, uint MatrixNum, const int flag);


        void re_construct(uint width, uint height, uint MatrixNum, const int flag);


        _MatrixArray();


        _MatrixArray(uint width, uint height, uint MatrixNum, const int flag);


        virtual uint Width() { return this->width; }


        virtual uint Height() { return this->height; }


        virtual uint MatrixNumber() { return this->ArrayNumber; }


        virtual T& index(uint row, uint col, size_t _seq);


        virtual de::MatrixArray<T>& operator=(de::MatrixArray<T>& src);


        virtual void release();
    };

}


template <typename T>
void decx::_MatrixArray<T>::alloc_data_space()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        if (decx::alloc::_host_virtual_page_malloc<T>(&this->MatArr, this->total_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for MatrixArray on host\n");
            return;
        }
        break;

#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_malloc<T>(&this->MatArr, this->total_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for MatrixArray on host\n");
            return;
        }
        break;
#endif

    default:
        break;
    }

    memset(this->MatArr.ptr, 0, this->total_bytes);

    if (decx::alloc::_host_virtual_page_malloc<T*>(&this->MatptrArr, this->ArrayNumber * sizeof(T*))) {
        Print_Error_Message(4, "Fail to allocate memory for pointer array on host\n");
        return;
    }
    this->MatptrArr.ptr[0] = this->MatArr.ptr;
    for (int i = 1; i < this->ArrayNumber; ++i) {
        this->MatptrArr.ptr[i] = this->MatptrArr.ptr[i - 1] + this->_plane;
    }
}



template <typename T>
void decx::_MatrixArray<T>::re_alloc_data_space()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        if (decx::alloc::_host_virtual_page_realloc<T>(&this->MatArr, this->total_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for MatrixArray on host\n");
            return;
        }
        break;

#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_realloc<T>(&this->MatArr, this->total_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for MatrixArray on host\n");
            return;
        }
        break;
#endif

    default:
        break;
    }

    memset(this->MatArr.ptr, 0, this->total_bytes);

    if (decx::alloc::_host_virtual_page_malloc<T*>(&this->MatptrArr, this->ArrayNumber * sizeof(T*))) {
        Print_Error_Message(4, "Fail to allocate memory for pointer array on host\n");
        return;
    }
    this->MatptrArr.ptr[0] = this->MatArr.ptr;
    for (int i = 1; i < this->ArrayNumber; ++i) {
        this->MatptrArr.ptr[i] = this->MatptrArr.ptr[i - 1] + this->_plane;
    }
}



template <typename T>
void decx::_MatrixArray<T>::construct(uint _width, uint height, uint _MatrixNum, const int _flag)
{
    this->_attribute_assign(_width, _height, _MatrixNum, _flag);

    this->alloc_data_space();
}


template <typename T>
void decx::_MatrixArray<T>::re_construct(uint _width, uint _height, uint _MatrixNum, const int _flag)
{
    // If all the parameters are the same, it is meaningless to re-construt the data
    if (this->width != _width || this->height != _height || this->ArrayNumber != _MatrixNum || this->_store_type != _flag) 
    {
        this->_attribute_assign(_width, _height, _MatrixNum, _flag);

        this->re_alloc_data_space();
    }
}



void decx::_MatrixArray<float>::_attribute_assign(uint _width, uint _height, uint MatrixNum, const int flag)
{
    this->width = _width;
    this->height = _height;
    this->ArrayNumber = MatrixNum;

    this->pitch = decx::utils::ceil<size_t>(_width, _MATRIX_ALIGN_4B_) * _MATRIX_ALIGN_4B_;

    this->plane = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->_plane = static_cast<size_t>(this->pitch) * static_cast<size_t>(this->height);

    this->element_num = this->plane * MatrixNum;
    this->_element_num = this->_plane * MatrixNum;

    this->total_bytes = this->_element_num * sizeof(float);
}



void decx::_MatrixArray<int>::_attribute_assign(uint _width, uint _height, uint MatrixNum, const int flag)
{
    this->width = _width;
    this->height = _height;
    this->ArrayNumber = MatrixNum;

    this->pitch = decx::utils::ceil<size_t>(_width, _MATRIX_ALIGN_4B_) * _MATRIX_ALIGN_4B_;

    this->plane = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->_plane = static_cast<size_t>(this->pitch) * static_cast<size_t>(this->height);

    this->element_num = this->plane * MatrixNum;
    this->_element_num = this->_plane * MatrixNum;

    this->total_bytes = this->_element_num * sizeof(int);
}



void decx::_MatrixArray<double>::_attribute_assign(uint _width, uint _height, uint MatrixNum, const int flag)
{
    this->width = _width;
    this->height = _height;
    this->ArrayNumber = MatrixNum;

    this->pitch = decx::utils::ceil<size_t>(_width, _MATRIX_ALIGN_8B_) * _MATRIX_ALIGN_8B_;

    this->plane = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->_plane = static_cast<size_t>(this->pitch) * static_cast<size_t>(this->height);

    this->element_num = this->plane * MatrixNum;
    this->_element_num = this->_plane * MatrixNum;

    this->total_bytes = this->_element_num * sizeof(double);
}


#ifdef _DECX_CUDA_CODES_
void decx::_MatrixArray<de::Half>::_attribute_assign(uint _width, uint _height, uint MatrixNum, const int flag)
{
    this->width = _width;
    this->height = _height;
    this->ArrayNumber = MatrixNum;

    this->pitch = decx::utils::ceil<size_t>(_width, _MATRIX_ALIGN_2B_) * _MATRIX_ALIGN_2B_;

    this->plane = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->_plane = static_cast<size_t>(this->pitch) * static_cast<size_t>(this->height);

    this->element_num = this->plane * MatrixNum;
    this->_element_num = this->_plane * MatrixNum;

    this->total_bytes = this->_element_num * sizeof(de::Half);
}
#endif


void decx::_MatrixArray<uchar>::_attribute_assign(uint _width, uint _height, uint MatrixNum, const int flag)
{
    this->width = _width;
    this->height = _height;
    this->ArrayNumber = MatrixNum;

    this->pitch = decx::utils::ceil<size_t>(_width, _MATRIX_ALIGN_1B_) * _MATRIX_ALIGN_1B_;

    this->plane = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->_plane = static_cast<size_t>(this->pitch) * static_cast<size_t>(this->height);

    this->element_num = this->plane * MatrixNum;
    this->_element_num = this->_plane * MatrixNum;

    this->total_bytes = this->_element_num * sizeof(uchar);
}



void decx::_MatrixArray<de::CPf>::_attribute_assign(uint _width, uint _height, uint MatrixNum, const int flag)
{
    this->width = _width;
    this->height = _height;
    this->ArrayNumber = MatrixNum;

    this->pitch = decx::utils::ceil<size_t>(_width, _MATRIX_ALIGN_8B_) * _MATRIX_ALIGN_8B_;

    this->plane = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->_plane = static_cast<size_t>(this->pitch) * static_cast<size_t>(this->height);

    this->element_num = this->plane * MatrixNum;
    this->_element_num = this->_plane * MatrixNum;

    this->total_bytes = this->_element_num * sizeof(de::CPf);
}



template <typename T>
decx::_MatrixArray<T>::_MatrixArray()
{
    this->_attribute_assign(0, 0, 0, 0);
}



template <typename T>
decx::_MatrixArray<T>::_MatrixArray(uint W, uint H, uint MatrixNum, const int flag)
{
    this->_attribute_assign(W, H, MatrixNum, flag);

    this->alloc_data_space();
}



template <typename T>
T& decx::_MatrixArray<T>::index(uint row, uint col, size_t _seq)
{
    return this->MatptrArr.ptr[_seq][row * this->pitch + col];
}



namespace de
{
    template <typename T>
    de::MatrixArray<T>& CreateMatrixArrayRef();


    template <typename T>
    de::MatrixArray<T>* CreateMatrixArrayPtr();


    template <typename T>
    de::MatrixArray<T>& CreateMatrixArrayRef(uint width, uint height, uint MatrixNum, const int flag);


    template <typename T>
    de::MatrixArray<T>* CreateMatrixArrayPtr(uint width, uint height, uint MatrixNum, const int flag);
}



template <typename T>
de::MatrixArray<T>& de::CreateMatrixArrayRef()
{
    return *(new decx::_MatrixArray<T>());
}

template _DECX_API_ de::MatrixArray<int>&        de::CreateMatrixArrayRef();
template _DECX_API_ de::MatrixArray<float>&        de::CreateMatrixArrayRef();
template _DECX_API_ de::MatrixArray<de::Half>&    de::CreateMatrixArrayRef();
template _DECX_API_ de::MatrixArray<double>&    de::CreateMatrixArrayRef();
template _DECX_API_ de::MatrixArray<de::CPf>&    de::CreateMatrixArrayRef();



template <typename T>
de::MatrixArray<T>* de::CreateMatrixArrayPtr()
{
    return new decx::_MatrixArray<T>();
}

template _DECX_API_ de::MatrixArray<int>*        de::CreateMatrixArrayPtr();
template _DECX_API_ de::MatrixArray<float>*        de::CreateMatrixArrayPtr();
template _DECX_API_ de::MatrixArray<de::Half>*    de::CreateMatrixArrayPtr();
template _DECX_API_ de::MatrixArray<double>*    de::CreateMatrixArrayPtr();
template _DECX_API_ de::MatrixArray<de::CPf>*    de::CreateMatrixArrayPtr();



template <typename T>
de::MatrixArray<T>& de::CreateMatrixArrayRef(uint width, uint height, uint MatrixNum, const int flag)
{
    return *(new decx::_MatrixArray<T>(width, height, MatrixNum, flag));
}

template _DECX_API_ de::MatrixArray<int>&        de::CreateMatrixArrayRef(uint width, uint height, uint MatrixNum, const int flag);
template _DECX_API_ de::MatrixArray<float>&        de::CreateMatrixArrayRef(uint width, uint height, uint MatrixNum, const int flag);
template _DECX_API_ de::MatrixArray<de::Half>&    de::CreateMatrixArrayRef(uint width, uint height, uint MatrixNum, const int flag);
template _DECX_API_ de::MatrixArray<double>&    de::CreateMatrixArrayRef(uint width, uint height, uint MatrixNum, const int flag);
template _DECX_API_ de::MatrixArray<de::CPf>&    de::CreateMatrixArrayRef(uint width, uint height, uint MatrixNum, const int flag);



template <typename T>
de::MatrixArray<T>* de::CreateMatrixArrayPtr(uint width, uint height, uint MatrixNum, const int flag)
{
    return new decx::_MatrixArray<T>(width, height, MatrixNum, flag);
}

template _DECX_API_ de::MatrixArray<int>*        de::CreateMatrixArrayPtr(uint width, uint height, uint MatrixNum, const int flag);
template _DECX_API_ de::MatrixArray<float>*        de::CreateMatrixArrayPtr(uint width, uint height, uint MatrixNum, const int flag);
template _DECX_API_ de::MatrixArray<de::Half>*    de::CreateMatrixArrayPtr(uint width, uint height, uint MatrixNum, const int flag);
template _DECX_API_ de::MatrixArray<double>*    de::CreateMatrixArrayPtr(uint width, uint height, uint MatrixNum, const int flag);
template _DECX_API_ de::MatrixArray<de::CPf>*    de::CreateMatrixArrayPtr(uint width, uint height, uint MatrixNum, const int flag);


template <typename T>
void decx::_MatrixArray<T>::release()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_dealloc(&this->MatArr);
        break;

#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_dealloc(&this->MatArr);
        break;
#endif

    default:
        break;
    }

    decx::alloc::_host_virtual_page_dealloc(&this->MatptrArr);
}


template <typename T>
de::MatrixArray<T>& decx::_MatrixArray<T>::operator=(de::MatrixArray<T>& src)
{
    const decx::_MatrixArray<T>& ref_src = dynamic_cast<decx::_MatrixArray<T>&>(src);

    this->MatArr.block = ref_src.MatArr.block;

    this->_attribute_assign(ref_src.width, ref_src.height, ref_src.ArrayNumber, ref_src._store_type);

    switch (ref_src._store_type)
    {
#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_malloc_same_place(&this->MatArr);
        break;
#endif

    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_malloc_same_place(&this->MatArr);
        break;

    default:
        break;
    }

    return *this;
}

#endif        // #ifndef _DECX_COMBINED_

#endif        // #ifndef _MATRIXARRAY_H_