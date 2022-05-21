/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _TENSOR_H_
#define _TENSOR_H_


#include "../core/basic.h"
#include "../core/allocators.h"
#include "../classes/classes_util.h"
#include "store_types.h"


#define _TENSOR_ALIGN_4B_ 4
#define _TENSOR_ALIGN_8B_ 2
#define _TENSOR_ALIGN_2B_ 8
#define _TENSOR_ALIGN_1B_ 16


namespace de
{
    /** 用 channel_adjcent 的方式存储，将 channel 的数量凑成8倍数, CPU端可以用__m256，GPU端可以用float2
    * 将两字节的 half 类型凑成四倍数,GPU端可以凑成偶倍的 half2, 将 double 等八字节的数据类型凑成偶数*/
    template <typename T>
    class _DECX_API_ Tensor
    {
    public:
        Tensor() {}


        virtual uint Width() = 0;


        virtual uint Height() = 0;


        virtual uint Depth() = 0;


        virtual T& index(const int x, const int y, const int z) = 0;


        virtual de::Tensor<T>& operator=(de::Tensor<T>& src) = 0;


        virtual void release() = 0;
    };
}


#ifndef _DECX_COMBINED_

/**
* The data storage structure is shown below
*            
*            <--------------------- dp_x_wp ------------------->
*            <--------------- width ----------------> 4x
*             <-dpitch->
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] T            
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] |            
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] |            
*            ...                                     ...   ...        |    height
*            ...                                     ...   ...        |            
*            ...                                     ...   ...        |            
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] _    
* 
* Where : the vector along depth-axis
*    <------------ dpitch ----------->
*    <---- pitch ------>
*    [x x x x x x x x x 0 0 0 0 0 0 0]
*/

namespace decx
{
    template<typename T>
    // z-channel stored adjacently
    class _Tensor : public de::Tensor<T>
    {
    public:
        uint width,
            height,
            depth;

        int _store_type;

        uint dpitch;            // NOT IN BYTES, the true depth (4x)
        uint wpitch;            // NOT IN BYTES, the true width (4x)
        size_t dp_x_wp;            // NOT IN BYTES, true depth multiply true width

        decx::PtrInfo<T> Tens;
        size_t element_num;        // is the number of all the ACTIVE elements
        size_t total_bytes;        // is the size of ALL(including pitch) elements

        /* 
         * is the number of ACTIVE elements on a xy, xz, yz-plane, 
         *  plane[0] : plane-WH
         *  plane[1] : plane-WD
         *  plane[2] : plane-HD
         */
        size_t plane[3];

        size_t _element_num;        // the total number of elements, including Non_active numbers


        _Tensor();


        void _attribute_assign(const uint _width, const uint _height, const uint _depth, const int store_type);


        _Tensor(const uint _width, const uint _height, const uint _depth, const int store_type);


        virtual T& index(const int x, const int y, const int z);


        virtual uint Width() { return this->width; }


        virtual uint Height() { return this->height; }


        virtual uint Depth() { return this->depth; }


        virtual de::Tensor<T>& operator=(de::Tensor<T>& src);


        virtual void release();
    };
}


template<typename T>
decx::_Tensor<T>::_Tensor()
{
    this->_attribute_assign(0, 0, 0, 0);
}


template<typename T>
decx::_Tensor<T>::_Tensor(const uint _width, const uint _height, const uint _depth, const int store_type)
{
    this->_attribute_assign(_width, _height, _depth, store_type);

    switch (store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_malloc<T>(&this->Tens, this->total_bytes);
        break;

    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_malloc<T>(&this->Tens, this->total_bytes);
        break;

    default:
        break;
    }

    memset(this->Tens.ptr, 0, this->total_bytes);
}



void decx::_Tensor<float>::_attribute_assign(const uint _width, const uint _height, const uint _depth, const int store_type)
{
    this->width = _width;
    this->height = _height;
    this->depth = _depth;

    this->_store_type = store_type;
    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;

    
    this->dpitch = decx::utils::ceil<uint>(_depth, _TENSOR_ALIGN_4B_) * _TENSOR_ALIGN_4B_;
        

    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);

    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);

    this->element_num = static_cast<size_t>(this->depth) * this->plane[0];
    this->_element_num = static_cast<size_t>(this->height) * this->dp_x_wp;
    this->total_bytes = this->_element_num * sizeof(float);
}



void decx::_Tensor<int>::_attribute_assign(const uint _width, const uint _height, const uint _depth, const int store_type)
{
    this->width = _width;
    this->height = _height;
    this->depth = _depth;

    this->_store_type = store_type;
    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;


    this->dpitch = decx::utils::ceil<uint>(_depth, _TENSOR_ALIGN_4B_) * _TENSOR_ALIGN_4B_;


    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);

    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);

    this->element_num = static_cast<size_t>(this->depth) * this->plane[0];
    this->_element_num = static_cast<size_t>(this->height) * this->dp_x_wp;
    this->total_bytes = this->_element_num * sizeof(int);
}


#ifdef _DECX_CUDA_CODES_
void decx::_Tensor<de::Half>::_attribute_assign(const uint _width, const uint _height, const uint _depth, const int store_type)
{
    this->width = _width;
    this->height = _height;
    this->depth = _depth;

    this->_store_type = store_type;
    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;


    this->dpitch = decx::utils::ceil<uint>(_depth, _TENSOR_ALIGN_2B_) * _TENSOR_ALIGN_2B_;


    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);

    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);

    this->element_num = static_cast<size_t>(this->depth) * this->plane[0];
    this->_element_num = static_cast<size_t>(this->height) * this->dp_x_wp;
    this->total_bytes = this->_element_num * sizeof(de::Half);
}
#endif


void decx::_Tensor<double>::_attribute_assign(const uint _width, const uint _height, const uint _depth, const int store_type)
{
    this->width = _width;
    this->height = _height;
    this->depth = _depth;

    this->_store_type = store_type;
    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;


    this->dpitch = decx::utils::ceil<uint>(_depth, _TENSOR_ALIGN_8B_) * _TENSOR_ALIGN_8B_;


    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);

    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);

    this->element_num = static_cast<size_t>(this->depth) * this->plane[0];
    this->_element_num = static_cast<size_t>(this->height) * this->dp_x_wp;
    this->total_bytes = this->_element_num * sizeof(double);
}



void decx::_Tensor<uchar>::_attribute_assign(const uint _width, const uint _height, const uint _depth, const int store_type)
{
    this->width = _width;
    this->height = _height;
    this->depth = _depth;

    this->_store_type = store_type;
    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;


    this->dpitch = decx::utils::ceil<uint>(_depth, _TENSOR_ALIGN_1B_) * _TENSOR_ALIGN_1B_;


    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);

    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);

    this->element_num = static_cast<size_t>(this->depth) * this->plane[0];
    this->_element_num = static_cast<size_t>(this->height) * this->dp_x_wp;
    this->total_bytes = this->_element_num * sizeof(uchar);
}



template <typename T>
void decx::_Tensor<T>::release()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_dealloc(&this->Tens);
        break;

    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_dealloc(&this->Tens);
        break;

    default:
        break;
    }
}



template<typename T>
// x : row, y : col, z : depth
T& decx::_Tensor<T>::index(const int x, const int y, const int z)
{
    return (this->Tens.ptr)[(size_t)x * this->dp_x_wp + (size_t)y * (size_t)this->dpitch + (size_t)z];
}



namespace de
{
    template <typename T>
    de::Tensor<T>* CreateTensorPtr();


    template <typename T>
    de::Tensor<T>& CreateTensorRef();


    template <typename T>
    de::Tensor<T>* CreateTensorPtr(const uint _width, const uint _height, const uint _depth, const int flag);


    template <typename T>
    de::Tensor<T>& CreateTensorRef(const uint _width, const uint _height, const uint _depth, const int flag);
}



template <typename T>
de::Tensor<T>& de::CreateTensorRef()
{
    return *(new decx::_Tensor<T>());
}


template <typename T>
de::Tensor<T>* de::CreateTensorPtr()
{
    return new decx::_Tensor<T>();
}

template _DECX_API_ de::Tensor<int>& de::CreateTensorRef();

template _DECX_API_ de::Tensor<float>& de::CreateTensorRef();

template _DECX_API_ de::Tensor<double>& de::CreateTensorRef();

template _DECX_API_ de::Tensor<uchar>& de::CreateTensorRef();

#ifdef _DECX_CUDA_CODES_
template _DECX_API_ de::Tensor<de::Half>& de::CreateTensorRef();
#endif



template _DECX_API_ de::Tensor<int>* de::CreateTensorPtr();

template _DECX_API_ de::Tensor<float>* de::CreateTensorPtr();

template _DECX_API_ de::Tensor<double>* de::CreateTensorPtr();

template _DECX_API_ de::Tensor<uchar>* de::CreateTensorPtr();

#ifdef _DECX_CUDA_CODES_
template _DECX_API_ de::Tensor<de::Half>* de::CreateTensorPtr();
#endif



template <typename T>
de::Tensor<T>& de::CreateTensorRef(const uint _width, const uint _height, const uint _depth, const int flag)
{
    return *(new decx::_Tensor<T>(_width, _height, _depth, flag));
}


template <typename T>
de::Tensor<T>* de::CreateTensorPtr(const uint _width, const uint _height, const uint _depth, const int flag)
{
    return new decx::_Tensor<T>(_width, _height, _depth, flag);
}

template _DECX_API_ de::Tensor<int>& de::CreateTensorRef(const uint _width, const uint _height, const uint _depth, const int flag);

template _DECX_API_ de::Tensor<float>& de::CreateTensorRef(const uint _width, const uint _height, const uint _depth, const int flag);

template _DECX_API_ de::Tensor<double>& de::CreateTensorRef(const uint _width, const uint _height, const uint _depth, const int flag);

template _DECX_API_ de::Tensor<uchar>& de::CreateTensorRef(const uint _width, const uint _height, const uint _depth, const int flag);

#ifdef _DECX_CUDA_CODES_
template _DECX_API_ de::Tensor<de::Half>& de::CreateTensorRef(const uint _width, const uint _height, const uint _depth, const int flag);
#endif



template _DECX_API_ de::Tensor<int>* de::CreateTensorPtr(const uint _width, const uint _height, const uint _depth, const int flag);

template _DECX_API_ de::Tensor<float>* de::CreateTensorPtr(const uint _width, const uint _height, const uint _depth, const int flag);

template _DECX_API_ de::Tensor<double>* de::CreateTensorPtr(const uint _width, const uint _height, const uint _depth, const int flag);

template _DECX_API_ de::Tensor<uchar>* de::CreateTensorPtr(const uint _width, const uint _height, const uint _depth, const int flag);

#ifdef _DECX_CUDA_CODES_
template _DECX_API_ de::Tensor<de::Half>* de::CreateTensorPtr(const uint _width, const uint _height, const uint _depth, const int flag);
#endif



template <typename T>
de::Tensor<T>& decx::_Tensor<T>::operator=(de::Tensor<T>& src)
{
    decx::_Tensor<T>& ref_src = dynamic_cast<decx::_Tensor<T> &>(src);

    this->Tens.block = ref_src.Tens.block;

    this->_attribute_assign(ref_src.width, ref_src.height, ref_src.depth, ref_src._store_type);
    
    switch (ref_src._store_type)
    {
#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_malloc_same_place(&this->Tens);
        break;
#endif

    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_malloc_same_place(&this->Tens);
        break;

    default:
        break;
    }

    return *this;
}

#endif    // #ifndef _DECX_COMBINED_

#endif