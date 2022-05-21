/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _GPU_MATRIXARRAY_H_
#define _GPU_MATRIXARRAY_H_

#include "../core/basic.h"
#include "../core/allocators.h"
#include "../classes/GPU_Matrix.h"
#include "../classes/MatrixArray.h"


namespace de
{
    template <typename T>
    class _DECX_API_ GPU_MatrixArray
    {
    public:
        uint ArrayNumber;


        virtual uint Width() = 0;


        virtual uint Height() = 0;


        virtual uint MatrixNumber() = 0;
        

        virtual void Load_from_host(de::MatrixArray<T> &src) = 0;


        virtual void Load_to_host(de::MatrixArray<T>& dst) = 0;


        virtual de::GPU_MatrixArray<T>& operator=(de::GPU_MatrixArray<T>& src) = 0;


        virtual void release() = 0;
    };
}


#ifndef _DECX_COMBINED_


namespace decx
{
    template <typename T>
    class _GPU_MatrixArray : public de::GPU_MatrixArray<T>
    {
        // call AFTER attributes are assigned !
        // Once called, the data space will be re-constructed unconditionally, according to the 
        // attributes, the previous data will be lost
        void re_alloc_data_space();

        // call AFTER attributes are assigned !
        // Once called, the data space will be constructed unconditionally, according to the 
        // attributes
        void alloc_data_space();


        void _attribute_assign(uint width, uint height, uint MatrixNum);

    public:
        decx::PtrInfo<T> MatArr;
        decx::PtrInfo<T*> MatptrArr;

        uint width, height;

        size_t element_num, _element_num,
            total_bytes,    // The real total bytes of the MatrixArray memory block, ATTENTION : NO '_' at the front
            ArrayNumber;    // The number of matrices that share the same sizes

        size_t plane, _plane;

        uint pitch,            // the true width (NOT IN BYTES)
            _height;        // the true height

        _GPU_MatrixArray();


        _GPU_MatrixArray(uint width, uint height, uint MatrixNum);


        void construct(uint width, uint height, uint MatrixNum);


        void re_construct(uint width, uint height, uint MatrixNum);


        virtual uint Width() { return this->width; }


        virtual uint Height() { return this->height; }


        virtual uint MatrixNumber() { return this->ArrayNumber; }


        virtual void Load_from_host(de::MatrixArray<T>& src);


        virtual void Load_to_host(de::MatrixArray<T>& dst);


        virtual de::GPU_MatrixArray<T>& operator=(de::GPU_MatrixArray<T>& src);


        virtual void release();
    };
}




template <typename T>
void decx::_GPU_MatrixArray<T>::re_alloc_data_space()
{
    if (decx::alloc::_device_realloc(&this->MatArr, this->total_bytes)) {
        Print_Error_Message(4, "de::GPU_MatrixArray<T>:: Fail to allocate memory\n");
        exit(-1);
    }

    cudaMemset(this->MatArr.ptr, 0, this->total_bytes);

    if (decx::alloc::_host_virtual_page_realloc<T*>(&this->MatptrArr, this->ArrayNumber * sizeof(T*))) {
        Print_Error_Message(4, "Fail to allocate memory for pointer array on host\n");
        return;
    }
    this->MatptrArr.ptr[0] = this->MatArr.ptr;
    for (int i = 1; i < this->ArrayNumber; ++i) {
        this->MatptrArr.ptr[i] = this->MatptrArr.ptr[i - 1] + this->_plane;
    }
}



template <typename T>
void decx::_GPU_MatrixArray<T>::alloc_data_space()
{
    if (decx::alloc::_device_malloc(&this->MatArr, this->total_bytes)) {
        Print_Error_Message(4, "de::GPU_MatrixArray<T>:: Fail to allocate memory\n");
        exit(-1);
    }

    cudaMemset(this->MatArr.ptr, 0, this->total_bytes);

    if (decx::alloc::_host_virtual_page_malloc<T*>(&this->MatptrArr, this->ArrayNumber * sizeof(T*))) {
        Print_Error_Message(4, "Fail to allocate memory for pointer array on host\n");
        return;
    }
    this->MatptrArr.ptr[0] = this->MatArr.ptr;
    for (int i = 1; i < this->ArrayNumber; ++i) {
        this->MatptrArr.ptr[i] = this->MatptrArr.ptr[i - 1] + this->_plane;
    }
}



//template <typename T>
//void decx::_GPU_MatrixArray<T>::_attribute_assign(uint _width, uint _height, uint MatrixNum)
//{
//    this->width = _width;
//    this->height = _height;
//    this->ArrayNumber = MatrixNum;
//
//    this->pitch = decx::utils::ceil<size_t>(_width, DEVICE_MATRIX_COL_ALIGN) * DEVICE_MATRIX_COL_ALIGN;
//
//    this->plane = static_cast<size_t>(_width) * static_cast<size_t>(_height);
//    this->_plane = static_cast<size_t>(this->pitch) * static_cast<size_t>(this->height);
//
//    this->element_num = this->plane * MatrixNum;
//    this->_element_num = this->_plane * MatrixNum;
//
//    this->total_bytes = this->_element_num * sizeof(T);
//}




void decx::_GPU_MatrixArray<float>::_attribute_assign(uint _width, uint _height, uint MatrixNum)
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


void decx::_GPU_MatrixArray<int>::_attribute_assign(uint _width, uint _height, uint MatrixNum)
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


#ifdef _DECX_CUDA_CODES_
void decx::_GPU_MatrixArray<de::Half>::_attribute_assign(uint _width, uint _height, uint MatrixNum)
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


void decx::_GPU_MatrixArray<double>::_attribute_assign(uint _width, uint _height, uint MatrixNum)
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



void decx::_GPU_MatrixArray<de::CPf>::_attribute_assign(uint _width, uint _height, uint MatrixNum)
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



void decx::_GPU_MatrixArray<uchar>::_attribute_assign(uint _width, uint _height, uint MatrixNum)
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



template <typename T>
void decx::_GPU_MatrixArray<T>::construct(uint _width, uint _height, uint MatrixNum)
{
    this->_attribute_assign(_width, _height, MatrixNum);

    this->alloc_data_space();
}


template <typename T>
void decx::_GPU_MatrixArray<T>::re_construct(uint _width, uint _height, uint MatrixNum)
{
    if (_width != this->width || _height != this->height || MatrixNum != this->ArrayNumber) {
        this->_attribute_assign(_width, _height, MatrixNum);

        this->re_alloc_data_space();
    }
}



template <typename T>
decx::_GPU_MatrixArray<T>::_GPU_MatrixArray()
{
    this->_attribute_assign(0, 0, 0);
}


template <typename T>
decx::_GPU_MatrixArray<T>::_GPU_MatrixArray(uint W, uint H, uint MatrixNum)
{
    this->_attribute_assign(W, H, MatrixNum);

    this->alloc_data_space();
}


template <typename T>
void decx::_GPU_MatrixArray<T>::Load_from_host(de::MatrixArray<T>& src)
{
    decx::_MatrixArray<T>* _src = dynamic_cast<_MatrixArray<T>*>(&src);
    
    for (int i = 0; i < this->ArrayNumber; ++i) {
        checkCudaErrors(cudaMemcpy2D(this->MatptrArr.ptr[i], this->pitch * sizeof(T),
            _src->MatptrArr.ptr[i], _src->pitch * sizeof(T), this->width * sizeof(T), this->height,
            cudaMemcpyHostToDevice));
    }
}


template <typename T>
void decx::_GPU_MatrixArray<T>::Load_to_host(de::MatrixArray<T>& dst)
{
    decx::_MatrixArray<T>* _dst = dynamic_cast<_MatrixArray<T>*>(&dst);

    for (int i = 0; i < this->ArrayNumber; ++i) {
        checkCudaErrors(cudaMemcpy2D(_dst->MatptrArr.ptr[i], _dst->pitch * sizeof(T),
            this->MatptrArr.ptr[i], this->pitch * sizeof(T), _dst->width * sizeof(T), _dst->height,
            cudaMemcpyDeviceToHost));
    }
}



template <typename T>
void decx::_GPU_MatrixArray<T>::release()
{
    decx::alloc::_device_dealloc(&this->MatArr);
}




namespace de
{
    template <typename T>
    de::GPU_MatrixArray<T>& CreateGPUMatrixArrayRef();


    template <typename T>
    de::GPU_MatrixArray<T>* CreateGPUMatrixArrayPtr();


    template <typename T>
    de::GPU_MatrixArray<T>& CreateGPUMatrixArrayRef(const uint _width, const uint _height, const uint _Mat_number);


    template <typename T>
    de::GPU_MatrixArray<T>* CreateGPUMatrixArrayPtr(const uint _width, const uint _height, const uint _Mat_number);
}



template <typename T>
de::GPU_MatrixArray<T>& de::CreateGPUMatrixArrayRef()
{
    return *(new decx::_GPU_MatrixArray<T>());
}

template _DECX_API_ de::GPU_MatrixArray<int>& de::CreateGPUMatrixArrayRef();

template _DECX_API_ de::GPU_MatrixArray<float>& de::CreateGPUMatrixArrayRef();

#ifdef _DECX_CUDA_CODES_
template _DECX_API_ de::GPU_MatrixArray<de::Half>& de::CreateGPUMatrixArrayRef();
#endif


template <typename T>
de::GPU_MatrixArray<T>* de::CreateGPUMatrixArrayPtr()
{
    return new decx::_GPU_MatrixArray<T>();
}


template _DECX_API_ de::GPU_MatrixArray<int>* de::CreateGPUMatrixArrayPtr();

template _DECX_API_ de::GPU_MatrixArray<float>* de::CreateGPUMatrixArrayPtr();

#ifdef _DECX_CUDA_CODES_
template _DECX_API_ de::GPU_MatrixArray<de::Half>* de::CreateGPUMatrixArrayPtr();
#endif


template <typename T>
de::GPU_MatrixArray<T>& de::CreateGPUMatrixArrayRef(const uint _width, const uint _height, const uint _Mat_number)
{
    return *(new decx::_GPU_MatrixArray<T>(_width, _height, _Mat_number));
}

template _DECX_API_ de::GPU_MatrixArray<int>& de::CreateGPUMatrixArrayRef(const uint _width, const uint _height, const uint _Mat_number);

template _DECX_API_ de::GPU_MatrixArray<float>& de::CreateGPUMatrixArrayRef(const uint _width, const uint _height, const uint _Mat_number);

#ifdef _DECX_CUDA_CODES_
template _DECX_API_ de::GPU_MatrixArray<de::Half>& de::CreateGPUMatrixArrayRef(const uint _width, const uint _height, const uint _Mat_number);
#endif

template <typename T>
de::GPU_MatrixArray<T>* de::CreateGPUMatrixArrayPtr(const uint _width, const uint _height, const uint _Mat_number)
{
    return new decx::_GPU_MatrixArray<T>(_width, _height, _Mat_number);
}

template _DECX_API_ de::GPU_MatrixArray<int>* de::CreateGPUMatrixArrayPtr(const uint _width, const uint _height, const uint _Mat_number);

template _DECX_API_ de::GPU_MatrixArray<float>* de::CreateGPUMatrixArrayPtr(const uint _width, const uint _height, const uint _Mat_number);

#ifdef _DECX_CUDA_CODES_
template _DECX_API_ de::GPU_MatrixArray<de::Half>* de::CreateGPUMatrixArrayPtr(const uint _width, const uint _height, const uint _Mat_number);
#endif


template <typename T>
de::GPU_MatrixArray<T>& decx::_GPU_MatrixArray<T>::operator=(de::GPU_MatrixArray<T>& src)
{
    const decx::_GPU_MatrixArray<T>& ref_src = dynamic_cast<decx::_GPU_MatrixArray<T>&>(src);

    this->_attribute_assign(ref_src.width, ref_src.height, ref_src.ArrayNumber);
    decx::alloc::_device_malloc_same_place(&this->MatArr);

    return *this;
}


#endif        //#ifndef _DECX_COMBINED_


#endif