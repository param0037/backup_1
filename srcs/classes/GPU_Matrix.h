/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _GPU_MATRIX_H_
#define _GPU_MATRIX_H_

#include "../core/basic.h"
#include "../core/allocators.h"


namespace de
{
    template<typename T>
    class _DECX_API_ GPU_Matrix
    {
    public:
        GPU_Matrix() {}


        virtual uint Width() = 0;


        virtual uint Height() = 0;


        virtual size_t TotalBytes() = 0;


        virtual void Load_from_host(de::Matrix<T>& src) = 0;


        virtual void Load_to_host(de::Matrix<T>& dst) = 0;


        virtual void release() = 0;


        virtual de::GPU_Matrix<T>& operator=(de::GPU_Matrix<T>& src) = 0;


        ~GPU_Matrix() {}
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
*            ...                                  ...  |    height
*            ...                                  ...  |
*            ...                                  ...  |
*            [[x x x x x x... ...    ... ...x x x....] _
*/


namespace decx
{
    template<typename T>
    class _GPU_Matrix : public de::GPU_Matrix<T>
    {
    private:
        // call AFTER attributes are assigned !
        void re_alloc_data_space();

        // call AFTER attributes are assigned !
        void alloc_data_space();


        void _attribute_assign(uint width, uint height);

    public:
        uint width, height;

        unsigned short Store_Type;

        size_t _element_num, element_num, total_bytes, _total_bytes;
        uint pitch;            // the true width (NOT IN BYTES), it is aligned with 4

        decx::PtrInfo<T> Mat;


        void construct(uint width, uint height);


        void re_construct(uint width, uint height);


        _GPU_Matrix();


        _GPU_Matrix(const uint _width, const uint _height);


        virtual uint Width() { return this->width; }


        virtual uint Height() { return this->height; }


        virtual size_t TotalBytes() { return this->total_bytes; }


        virtual void Load_from_host(de::Matrix<T>& src);


        virtual void Load_to_host(de::Matrix<T>& dst);


        virtual void release();


        virtual de::GPU_Matrix<T>& operator=(de::GPU_Matrix<T>& src);


        ~_GPU_Matrix() {}
    };
}


//
//#ifndef DEVICE_MATRIX_COL_ALIGN
//#define DEVICE_MATRIX_COL_ALIGN 8
//#endif


template <typename T>
void decx::_GPU_Matrix<T>::alloc_data_space()
{
    if (decx::alloc::_device_malloc(&this->Mat, this->_total_bytes)) {
        SetConsoleColor(4);
        printf("Matrix malloc failed! Please check if there is enough space in your device.");
        ResetConsoleColor;
        return;
    }

    cudaMemset(this->Mat.ptr, 0, this->total_bytes);
}



template <typename T>
void decx::_GPU_Matrix<T>::re_alloc_data_space()
{
    if (decx::alloc::_device_realloc(&this->Mat, this->_total_bytes)) {
        SetConsoleColor(4);
        printf("Matrix malloc failed! Please check if there is enough space in your device.");
        ResetConsoleColor;
        return;
    }

    cudaMemset(this->Mat.ptr, 0, this->total_bytes);
}



template <typename T>
void decx::_GPU_Matrix<T>::construct(uint _width, uint _height)
{
    this->_attribute_assign(_width, _height);

    this->alloc_data_space();
}


template <typename T>
void decx::_GPU_Matrix<T>::re_construct(uint _width, uint _height)
{
    // If all the parameters are the same, it is meaningless to re-construt the data
    if (this->width != _width || this->height != _height)
    {
        this->_attribute_assign(_width, _height);

        this->re_alloc_data_space();
    }
}


//template <typename T>
//void decx::_GPU_Matrix<T>::_attribute_assign(const uint _width, const uint _height)
//{
//    this->width = _width;
//    this->height = _height;
//
//    this->pitch = decx::utils::ceil<int>(_width, DEVICE_MATRIX_COL_ALIGN) * DEVICE_MATRIX_COL_ALIGN;
//
//    this->element_num = static_cast<size_t>(_width) * static_cast<size_t>(_height);
//    this->total_bytes = (this->element_num) * __SPACE__;
//
//    this->_element_num = static_cast<size_t>(this->pitch) * static_cast<size_t>(height);
//    this->_total_bytes = (this->_element_num) * __SPACE__;
//}



void decx::_GPU_Matrix<float>::_attribute_assign(const uint _width, const uint _height)
{
    this->width = _width;
    this->height = _height;

    this->pitch = decx::utils::ceil<int>(_width, _MATRIX_ALIGN_4B_) * _MATRIX_ALIGN_4B_;

    this->element_num = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->total_bytes = (this->element_num) * sizeof(float);

    this->_element_num = static_cast<size_t>(this->pitch) * static_cast<size_t>(this->height);
    this->_total_bytes = (this->_element_num) * sizeof(float);
}



void decx::_GPU_Matrix<int>::_attribute_assign(const uint _width, const uint _height)
{
    this->width = _width;
    this->height = _height;

    this->pitch = decx::utils::ceil<int>(_width, _MATRIX_ALIGN_4B_) * _MATRIX_ALIGN_4B_;

    this->element_num = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->total_bytes = (this->element_num) * sizeof(int);

    this->_element_num = static_cast<size_t>(this->pitch) * static_cast<size_t>(this->height);
    this->_total_bytes = (this->_element_num) * sizeof(int);
}



void decx::_GPU_Matrix<double>::_attribute_assign(const uint _width, const uint _height)
{
    this->width = _width;
    this->height = _height;

    this->pitch = decx::utils::ceil<int>(_width, _MATRIX_ALIGN_8B_) * _MATRIX_ALIGN_8B_;

    this->element_num = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->total_bytes = (this->element_num) * sizeof(double);

    this->_element_num = static_cast<size_t>(this->pitch) * static_cast<size_t>(this->height);
    this->_total_bytes = (this->_element_num) * sizeof(double);
}


#ifdef _DECX_CUDA_CODES_
void decx::_GPU_Matrix<de::Half>::_attribute_assign(const uint _width, const uint _height)
{
    this->width = _width;
    this->height = _height;

    this->pitch = decx::utils::ceil<int>(_width, _MATRIX_ALIGN_2B_) * _MATRIX_ALIGN_2B_;

    this->element_num = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->total_bytes = (this->element_num) * sizeof(de::Half);

    this->_element_num = static_cast<size_t>(this->pitch) * static_cast<size_t>(this->height);
    this->_total_bytes = (this->_element_num) * sizeof(de::Half);
}
#endif



void decx::_GPU_Matrix<de::CPf>::_attribute_assign(const uint _width, const uint _height)
{
    this->width = _width;
    this->height = _height;

    this->pitch = decx::utils::ceil<int>(_width, _MATRIX_ALIGN_8B_) * _MATRIX_ALIGN_8B_;

    this->element_num = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->total_bytes = (this->element_num) * sizeof(de::CPf);

    this->_element_num = static_cast<size_t>(this->pitch) * static_cast<size_t>(this->height);
    this->_total_bytes = (this->_element_num) * sizeof(de::CPf);
}



void decx::_GPU_Matrix<uchar>::_attribute_assign(const uint _width, const uint _height)
{
    this->width = _width;
    this->height = _height;

    this->pitch = decx::utils::ceil<int>(_width, _MATRIX_ALIGN_1B_) * _MATRIX_ALIGN_1B_;

    this->element_num = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->total_bytes = (this->element_num) * sizeof(uchar);

    this->_element_num = static_cast<size_t>(this->pitch) * static_cast<size_t>(height);
    this->_total_bytes = (this->_element_num) * sizeof(uchar);
}



template <typename T>
decx::_GPU_Matrix<T>::_GPU_Matrix(const uint _width, const uint _height)
{
    this->_attribute_assign(_width, _height);

    this->alloc_data_space();
}



template <typename T>
decx::_GPU_Matrix<T>::_GPU_Matrix()
{
    this->_attribute_assign(0, 0);
}



template <typename T>
void decx::_GPU_Matrix<T>::release()
{
    decx::alloc::_device_dealloc(&this->Mat);
}



template <typename T>
void decx::_GPU_Matrix<T>::Load_from_host(de::Matrix<T>& src)
{
    decx::_Matrix<T>* _src = dynamic_cast<decx::_Matrix<T>*>(&src);

    checkCudaErrors(cudaMemcpy2D(this->Mat.ptr, this->pitch * sizeof(T),
        _src->Mat.ptr, _src->pitch * sizeof(T), this->width * sizeof(T), this->height,
        cudaMemcpyHostToDevice));
}



template <typename T>
void decx::_GPU_Matrix<T>::Load_to_host(de::Matrix<T>& dst)
{
    decx::_Matrix<T>* _dst = dynamic_cast<decx::_Matrix<T>*>(&dst);

    checkCudaErrors(cudaMemcpy2D(_dst->Mat.ptr, _dst->pitch * sizeof(T),
        this->Mat.ptr, this->pitch * sizeof(T), _dst->width * sizeof(T), _dst->height,
        cudaMemcpyDeviceToHost));
}



namespace de
{
    template <typename T>
    _DECX_API_ de::GPU_Matrix<T>& CreateGPUMatrixRef();


    template <typename T>
    _DECX_API_ de::GPU_Matrix<T>* CreateGPUMatrixPtr();


    template <typename T>
    _DECX_API_ de::GPU_Matrix<T>& CreateGPUMatrixRef(const uint width, const uint height);

    template <typename T>
    _DECX_API_ de::GPU_Matrix<T>* CreateGPUMatrixPtr(const uint width, const uint height);
}



template <typename T>
de::GPU_Matrix<T>& de::CreateGPUMatrixRef()
{
    return *(new decx::_GPU_Matrix<T>());
}

template _DECX_API_ de::GPU_Matrix<int>& de::CreateGPUMatrixRef();
template _DECX_API_ de::GPU_Matrix<float>& de::CreateGPUMatrixRef();
template _DECX_API_ de::GPU_Matrix<double>& de::CreateGPUMatrixRef();
#ifdef _DECX_CUDA_CODES_
template _DECX_API_ de::GPU_Matrix<de::Half>& de::CreateGPUMatrixRef();
#endif
template _DECX_API_ de::GPU_Matrix<uchar>& de::CreateGPUMatrixRef();


template <typename T>
de::GPU_Matrix<T>* de::CreateGPUMatrixPtr()
{
    return new decx::_GPU_Matrix<T>();
}

template _DECX_API_ de::GPU_Matrix<int>* de::CreateGPUMatrixPtr();
template _DECX_API_ de::GPU_Matrix<float>* de::CreateGPUMatrixPtr();
template _DECX_API_ de::GPU_Matrix<double>* de::CreateGPUMatrixPtr();
#ifdef _DECX_CUDA_CODES_
template _DECX_API_ de::GPU_Matrix<de::Half>* de::CreateGPUMatrixPtr();
#endif
template _DECX_API_ de::GPU_Matrix<uchar>* de::CreateGPUMatrixPtr();



template <typename T>
de::GPU_Matrix<T>& de::CreateGPUMatrixRef(const uint width, const uint height)
{
    return *(new decx::_GPU_Matrix<T>(width, height));
}

template _DECX_API_ de::GPU_Matrix<int>& de::CreateGPUMatrixRef(const uint width, const uint height);
template _DECX_API_ de::GPU_Matrix<float>& de::CreateGPUMatrixRef(const uint width, const uint height);
template _DECX_API_ de::GPU_Matrix<double>& de::CreateGPUMatrixRef(const uint width, const uint height);
#ifdef _DECX_CUDA_CODES_
template _DECX_API_ de::GPU_Matrix<de::Half>& de::CreateGPUMatrixRef(const uint width, const uint height);
#endif
template _DECX_API_ de::GPU_Matrix<uchar>& de::CreateGPUMatrixRef(const uint width, const uint height);


template <typename T>
de::GPU_Matrix<T>* de::CreateGPUMatrixPtr(const uint width, const uint height)
{
    return new decx::_GPU_Matrix<T>(width, height);
}

template _DECX_API_ de::GPU_Matrix<int>* de::CreateGPUMatrixPtr(const uint width, const uint height);
template _DECX_API_ de::GPU_Matrix<float>* de::CreateGPUMatrixPtr(const uint width, const uint height);
template _DECX_API_ de::GPU_Matrix<double>* de::CreateGPUMatrixPtr(const uint width, const uint height);
#ifdef _DECX_CUDA_CODES_
template _DECX_API_ de::GPU_Matrix<de::Half>* de::CreateGPUMatrixPtr(const uint width, const uint height);
#endif
template _DECX_API_ de::GPU_Matrix<uchar>* de::CreateGPUMatrixPtr(const uint width, const uint height);




template <typename T>
de::GPU_Matrix<T>& decx::_GPU_Matrix<T>::operator=(de::GPU_Matrix<T>& src)
{
    decx::_GPU_Matrix<T>& ref_src = dynamic_cast<decx::_GPU_Matrix<T>&>(src);

    this->Mat.block = ref_src.Mat.block;

    this->_attribute_assign(ref_src.width, ref_src.height);
    decx::alloc::_device_malloc_same_place(&this->Mat);

    return *this;
}

#endif


#endif