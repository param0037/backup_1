/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _GPU_VECTOR_H_
#define _GPU_VECTOR_H_

#include "../core/basic.h"
#include "../core/allocators.h"


namespace de
{
    template <typename T>
    class _DECX_API_ GPU_Vector
    {
    public:
        GPU_Vector() {}


        virtual size_t Len() = 0;


        virtual void load_from_host(de::Vector<T>& src) = 0;


        virtual void load_to_host(de::Vector<T>& dst) = 0;


        virtual void release() = 0;


        virtual de::GPU_Vector<T>& operator=(de::GPU_Vector<T>& src) = 0;


        ~GPU_Vector() {}
    };
}


#ifndef _DECX_COMBINED_

namespace decx
{
    template <typename T>
    class _GPU_Vector : public de::GPU_Vector<T>
    {
    public:
        size_t length,
            _length,    // It is aligned with 4
            total_bytes;

        decx::PtrInfo<T> Vec;


        _GPU_Vector();


        void _attribute_assign(size_t length);


        _GPU_Vector(size_t length);


        virtual size_t Len() { return this->length; }


        virtual void load_from_host(de::Vector<T>& src);


        virtual void load_to_host(de::Vector<T>& dst);

        
        virtual void release();


        virtual de::GPU_Vector<T>& operator=(de::GPU_Vector<T>& src);


        ~_GPU_Vector() {}
    };
}


template <typename T>
void decx::_GPU_Vector<T>::_attribute_assign(size_t length)
{
    this->length = length;
    switch (sizeof(T))
    {
    case 4:
        this->_length = decx::utils::ceil<size_t>(length, (size_t)_VECTOR_ALIGN_4B_) * (size_t)_VECTOR_ALIGN_4B_;
        break;

    case 8:
        this->_length = decx::utils::ceil<size_t>(length, (size_t)_VECTOR_ALIGN_8B_) * (size_t)_VECTOR_ALIGN_8B_;
        break;

    case 2:
        this->_length = decx::utils::ceil<size_t>(length, (size_t)_VECTOR_ALIGN_2B_) * (size_t)_VECTOR_ALIGN_2B_;
        break;
    }
    
    this->total_bytes = this->_length * sizeof(T);
}



template <typename T>
decx::_GPU_Vector<T>::_GPU_Vector(size_t length)
{
    this->_attribute_assign(length);

    if (decx::alloc::_device_malloc(&this->Vec, this->total_bytes)) {
        SetConsoleColor(4);
        printf("Vector on GPU malloc failed! Please check if there is enough space in your device.");
        ResetConsoleColor;
        return;
    }
}


template <typename T>
decx::_GPU_Vector<T>::_GPU_Vector()
{
    this->_attribute_assign(0);
}


template <typename T>
void decx::_GPU_Vector<T>::load_from_host(de::Vector<T>& src)
{
    decx::_Vector<T>* _src = dynamic_cast<decx::_Vector<T>*>(&src);
    checkCudaErrors(cudaMemcpy(this->Vec.ptr, _src->Vec.ptr, this->length * sizeof(T), cudaMemcpyHostToDevice));
}


template <typename T>
void decx::_GPU_Vector<T>::load_to_host(de::Vector<T>& dst)
{
    decx::_Vector<T>* _dst = dynamic_cast<decx::_Vector<T>*>(&dst);
    checkCudaErrors(cudaMemcpy(_dst->Vec.ptr, this->Vec.ptr, this->length * sizeof(T), cudaMemcpyDeviceToHost));
}



template <typename T>
void decx::_GPU_Vector<T>::release()
{
    decx::alloc::_device_dealloc(&this->Vec);
}



template <typename T>
de::GPU_Vector<T>& decx::_GPU_Vector<T>::operator=(de::GPU_Vector<T>& src)
{
    const decx::_GPU_Vector<T>& ref_src = dynamic_cast<decx::_GPU_Vector<T>&>(src);

    this->_attribute_assign(ref_src.length);
    decx::alloc::_device_malloc_same_place(&this->Vec);

    return *this;
}



namespace de
{
    template <typename T>
    de::GPU_Vector<T>& CreateGPUVectorRef();


    template <typename T>
    de::GPU_Vector<T>* CreateGPUVectorPtr();


    template <typename T>
    de::GPU_Vector<T>& CreateGPUVectorRef(const size_t length);


    template <typename T>
    de::GPU_Vector<T>* CreateGPUVectorPtr(const size_t length);
}



template <typename T>
de::GPU_Vector<T>& de::CreateGPUVectorRef() {
    return *(new decx::_GPU_Vector<T>());
}
template _DECX_API_ de::GPU_Vector<int>& de::CreateGPUVectorRef();

template _DECX_API_ de::GPU_Vector<float>& de::CreateGPUVectorRef();

template _DECX_API_ de::GPU_Vector<double>& de::CreateGPUVectorRef();

template _DECX_API_ de::GPU_Vector<de::Half>& de::CreateGPUVectorRef();


template <typename T>
de::GPU_Vector<T>* de::CreateGPUVectorPtr() {
    return new decx::_GPU_Vector<T>();
}

template _DECX_API_ de::GPU_Vector<int>* de::CreateGPUVectorPtr<int>();

template _DECX_API_ de::GPU_Vector<float>* de::CreateGPUVectorPtr<float>();

template _DECX_API_ de::GPU_Vector<double>* de::CreateGPUVectorPtr<double>();

template _DECX_API_ de::GPU_Vector<de::Half>* de::CreateGPUVectorPtr();





template <typename T>
de::GPU_Vector<T>& de::CreateGPUVectorRef(const size_t length) {
    return *(new decx::_GPU_Vector<T>(length));
}
template _DECX_API_ de::GPU_Vector<int>& de::CreateGPUVectorRef(const size_t length);

template _DECX_API_ de::GPU_Vector<float>& de::CreateGPUVectorRef(const size_t length);

template _DECX_API_ de::GPU_Vector<double>& de::CreateGPUVectorRef(const size_t length);

template _DECX_API_ de::GPU_Vector<de::Half>& de::CreateGPUVectorRef(const size_t length);


template <typename T>
de::GPU_Vector<T>* de::CreateGPUVectorPtr(const size_t length) {
    return new decx::_GPU_Vector<T>(length);
}

template _DECX_API_ de::GPU_Vector<int>* de::CreateGPUVectorPtr<int>(const size_t length);

template _DECX_API_ de::GPU_Vector<float>* de::CreateGPUVectorPtr<float>(const size_t length);

template _DECX_API_ de::GPU_Vector<double>* de::CreateGPUVectorPtr<double>(const size_t length);

template _DECX_API_ de::GPU_Vector<de::Half>* de::CreateGPUVectorPtr(const size_t length);

#endif        //#ifndef _DECX_COMBINED_

#endif