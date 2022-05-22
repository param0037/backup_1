/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _TENSORARRAY_H_
#define _TENSORARRAY_H_


#include "../core/basic.h"
#include "../core/allocators.h"
#include "../classes/classes_util.h"
#include "store_types.h"
#include "tensor.h"


namespace de
{
    template <typename T>
    class _DECX_API_ TensorArray
    {
    public:
        TensorArray() {}


        virtual uint Width() = 0;


        virtual uint Height() = 0;


        virtual uint Depth() = 0;


        virtual uint TensorNum() = 0;


        virtual T& index(const int x, const int y, const int z, const int tensor_id) = 0;


        virtual de::TensorArray<T>& operator=(de::TensorArray<T>& src) = 0;


        virtual void release() = 0;
    };
}


#ifndef _DECX_COMBINED_

/**
* The data storage structure is shown below
* tensor_id
*            <-------------------- dp_x_w --------------------->
*             <---------------- width -------------->
*             <-dpitch->
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] T            T
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] |            |
*    0       [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] |            |
*            ...                                     ...         |    height  |    hpitch(2x)
*            ...                                     ...         |            |
*            ...                                     ...         |            |
*       ___> [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] _            _
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]]
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]]
*    1       [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]]
*            ...                                     ...
*            ...                                     ...
*            ...                                     ...
*       ___> [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]]
*    .
*    .
*    .
*
* Where : the vector along depth-axis
*    <------------ dpitch ----------->
*    <---- pitch ------>
*    [x x x x x x x x x 0 0 0 0 0 0 0]
*/

namespace decx
{
    template <typename T>
    class _TensorArray : public de::TensorArray<T>
    {
    private:
        void _attribute_assign(const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int store_type);


        void alloc_data_space();


        void re_alloc_data_space();

    public:
        uint width,
             height,
             depth,
             tensor_num;

        int _store_type;

        // The data pointer
        decx::PtrInfo<T> TensArr;
        // The pointer array for the pointers of each tensor in the TensorArray
        decx::PtrInfo<T*> TensptrArr;

        uint dpitch;            // NOT IN BYTES, the true depth (4x)
        uint wpitch;            // NOT IN BYTES, the true width (2x)
        size_t dp_x_wp;            // NOT IN BYTES, true depth multiply true width

        /*
         * is the number of ACTIVE elements on a xy, xz, yz-plane,
         *  plane[0] : plane-WH
         *  plane[1] : plane-WD
         *  plane[2] : plane-HD
         */
        size_t plane[3];

        // The true size of a Tensor, including pitch
        size_t _gap;

        // The number of all the active elements in the TensorArray
        size_t element_num;

        // The number of all the elements in the TensorArray, including pitch
        size_t _element_num;

        // The size of all the elements in the TensorArray, including pitch
        size_t total_bytes;


        _TensorArray();


        _TensorArray(const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int store_type);


        void construct(const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int flag);


        void re_construct(const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int flag);


        virtual uint Width() { return this->width; }


        virtual uint Height() { return this->height; }


        virtual uint Depth() { return this->depth; }


        virtual uint TensorNum() { return this->tensor_num; }


        virtual T& index(const int x, const int y, const int z, const int tensor_id);


        virtual de::TensorArray<T>& operator=(de::TensorArray<T>& src);


        virtual void release();
    };
}



void decx::_TensorArray<float>::_attribute_assign(const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int store_type)
{
    this->width = _width;
    this->height = _height;
    this->depth = _depth;
    this->tensor_num = _tensor_num;

    this->_store_type = store_type;
    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;


    this->dpitch = decx::utils::ceil<uint>(_depth, _TENSOR_ALIGN_4B_) * _TENSOR_ALIGN_4B_;


    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);

    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);

    this->_gap = this->dp_x_wp * static_cast<size_t>(this->height);

    this->element_num = static_cast<size_t>(this->depth) * this->plane[0] * static_cast<size_t>(this->tensor_num);
    this->_element_num = static_cast<size_t>(this->tensor_num) * this->_gap;
    this->total_bytes = this->_element_num * sizeof(float);
}



void decx::_TensorArray<int>::_attribute_assign(const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int store_type)
{
    this->width = _width;
    this->height = _height;
    this->depth = _depth;
    this->tensor_num = _tensor_num;

    this->_store_type = store_type;
    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;


    this->dpitch = decx::utils::ceil<uint>(_depth, _TENSOR_ALIGN_4B_) * _TENSOR_ALIGN_4B_;


    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);

    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);

    this->_gap = this->dp_x_wp * static_cast<size_t>(this->height);

    this->element_num = static_cast<size_t>(this->depth) * this->plane[0] * static_cast<size_t>(this->tensor_num);
    this->_element_num = static_cast<size_t>(this->tensor_num) * this->_gap;
    this->total_bytes = this->_element_num * sizeof(int);
}



#ifndef GNU_CPUcodes
void decx::_TensorArray<de::Half>::_attribute_assign(const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int store_type)
{
    this->width = _width;
    this->height = _height;
    this->depth = _depth;
    this->tensor_num = _tensor_num;

    this->_store_type = store_type;
    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;


    this->dpitch = decx::utils::ceil<uint>(_depth, _TENSOR_ALIGN_2B_) * _TENSOR_ALIGN_2B_;


    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);

    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);

    this->_gap = this->dp_x_wp * static_cast<size_t>(this->height);

    this->element_num = static_cast<size_t>(this->depth) * this->plane[0] * static_cast<size_t>(this->tensor_num);
    this->_element_num = static_cast<size_t>(this->tensor_num) * this->_gap;
    this->total_bytes = this->_element_num * sizeof(de::Half);
}
#endif        // #ifndef GNU_CPUcodes



void decx::_TensorArray<uchar>::_attribute_assign(const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int store_type)
{
    this->width = _width;
    this->height = _height;
    this->depth = _depth;
    this->tensor_num = _tensor_num;

    this->_store_type = store_type;
    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;


    this->dpitch = decx::utils::ceil<uint>(_depth, _TENSOR_ALIGN_1B_) * _TENSOR_ALIGN_1B_;


    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);

    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);

    this->_gap = this->dp_x_wp * static_cast<size_t>(this->height);

    this->element_num = static_cast<size_t>(this->depth) * this->plane[0] * static_cast<size_t>(this->tensor_num);
    this->_element_num = static_cast<size_t>(this->tensor_num) * this->_gap;
    this->total_bytes = this->_element_num * sizeof(uchar);
}



void decx::_TensorArray<double>::_attribute_assign(const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int store_type)
{
    this->width = _width;
    this->height = _height;
    this->depth = _depth;
    this->tensor_num = _tensor_num;

    this->_store_type = store_type;
    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;


    this->dpitch = decx::utils::ceil<uint>(_depth, _TENSOR_ALIGN_8B_) * _TENSOR_ALIGN_8B_;


    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);

    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);

    this->_gap = this->dp_x_wp * static_cast<size_t>(this->height);

    this->element_num = static_cast<size_t>(this->depth) * this->plane[0] * static_cast<size_t>(this->tensor_num);
    this->_element_num = static_cast<size_t>(this->tensor_num) * this->_gap;
    this->total_bytes = this->_element_num * sizeof(double);
}



template <typename T>
void decx::_TensorArray<T>::alloc_data_space()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_malloc<T>(&this->TensArr, this->total_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
            exit(-1);
        }
        break;

    case decx::DATA_STORE_TYPE::Page_Default:
        if (decx::alloc::_host_virtual_page_malloc<T>(&this->TensArr, this->total_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
            exit(-1);
        }
        break;

    default:
        break;
    }

    memset(this->TensArr.ptr, 0, this->total_bytes);

    if (decx::alloc::_host_virtual_page_malloc<T*>(&this->TensptrArr, this->tensor_num * sizeof(T*))) {
        Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
        return;
    }
    this->TensptrArr.ptr[0] = this->TensArr.ptr;
    for (uint i = 1; i < this->tensor_num; ++i) {
        this->TensptrArr.ptr[i] = this->TensptrArr.ptr[i - 1] + this->_gap;
    }
}



template <typename T>
void decx::_TensorArray<T>::re_alloc_data_space()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_realloc<T>(&this->TensArr, this->total_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
            exit(-1);
        }
        break;

    case decx::DATA_STORE_TYPE::Page_Default:
        if (decx::alloc::_host_virtual_page_realloc<T>(&this->TensArr, this->total_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
            exit(-1);
        }
        break;

    default:
        break;
    }

    memset(this->TensArr.ptr, 0, this->total_bytes);

    if (decx::alloc::_host_virtual_page_realloc<T*>(&this->TensptrArr, this->tensor_num * sizeof(T*))) {
        Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
        return;
    }
    this->TensptrArr.ptr[0] = this->TensArr.ptr;
    for (uint i = 1; i < this->tensor_num; ++i) {
        this->TensptrArr.ptr[i] = this->TensptrArr.ptr[i - 1] + this->_gap;
    }
}



template <typename T>
void decx::_TensorArray<T>::construct(const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int store_type)
{
    this->_attribute_assign(_width, _height, _depth, _tensor_num, store_type);

    this->alloc_data_space();
}



template <typename T>
void decx::_TensorArray<T>::re_construct(const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int store_type)
{
    if (this->width != _width || this->height != _height || this->depth != _depth || 
        this->tensor_num != _tensor_num || this->_store_type != store_type) 
    {
        this->_attribute_assign(_width, _height, _depth, _tensor_num, store_type);

        this->re_alloc_data_space();
    }
}



template<typename T>
decx::_TensorArray<T>::_TensorArray()
{
    this->_attribute_assign(0, 0, 0, 0, 0);
}



template<typename T>
decx::_TensorArray<T>::_TensorArray(const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int store_type)
{
    this->_attribute_assign(_width, _height, _depth, _tensor_num, store_type);

    this->alloc_data_space();
}


namespace de
{
    template <typename T>
    de::TensorArray<T>& CreateTensorArrayRef();


    template <typename T>
    de::TensorArray<T>* CreateTensorArrayPtr();


    template <typename T>
    de::TensorArray<T>& CreateTensorArrayRef(const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type);


    template <typename T>
    de::TensorArray<T>* CreateTensorArrayPtr(const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type);
}


template <typename T>
de::TensorArray<T>& de::CreateTensorArrayRef()
{
    return *(new decx::_TensorArray<T>());
}



template <typename T>
de::TensorArray<T>* de::CreateTensorArrayPtr()
{
    return new decx::_TensorArray<T>();
}



template <typename T>
de::TensorArray<T>& de::CreateTensorArrayRef(const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type)
{
    return *(new decx::_TensorArray<T>(width, height, depth, tensor_num, store_type));
}



template <typename T>
de::TensorArray<T>* de::CreateTensorArrayPtr(const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type)
{
    return new decx::_TensorArray<T>(width, height, depth, tensor_num, store_type);
}


template _DECX_API_ de::TensorArray<int>&        de::CreateTensorArrayRef();
template _DECX_API_ de::TensorArray<float>&        de::CreateTensorArrayRef();
template _DECX_API_ de::TensorArray<double>&    de::CreateTensorArrayRef();
#ifndef GNU_CPUcodes
template _DECX_API_ de::TensorArray<de::Half>&  de::CreateTensorArrayRef();
#endif
template _DECX_API_ de::TensorArray<uchar>&        de::CreateTensorArrayRef();


template _DECX_API_ de::TensorArray<int>*        de::CreateTensorArrayPtr();
template _DECX_API_ de::TensorArray<float>*        de::CreateTensorArrayPtr();
template _DECX_API_ de::TensorArray<double>*    de::CreateTensorArrayPtr();
#ifndef GNU_CPUcodes
template _DECX_API_ de::TensorArray<de::Half>*    de::CreateTensorArrayPtr();
#endif
template _DECX_API_ de::TensorArray<uchar>*        de::CreateTensorArrayPtr();


template _DECX_API_ de::TensorArray<int>&        de::CreateTensorArrayRef(const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type);
template _DECX_API_ de::TensorArray<float>&        de::CreateTensorArrayRef(const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type);
template _DECX_API_ de::TensorArray<double>&    de::CreateTensorArrayRef(const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type);
#ifndef GNU_CPUcodes
template _DECX_API_ de::TensorArray<de::Half>&    de::CreateTensorArrayRef(const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type);
#endif
template _DECX_API_ de::TensorArray<uchar>&        de::CreateTensorArrayRef(const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type);


template _DECX_API_ de::TensorArray<int>*        de::CreateTensorArrayPtr(const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type);
template _DECX_API_ de::TensorArray<float>*        de::CreateTensorArrayPtr(const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type);
template _DECX_API_ de::TensorArray<double>*    de::CreateTensorArrayPtr(const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type);
#ifndef GNU_CPUcodes
template _DECX_API_ de::TensorArray<de::Half>*    de::CreateTensorArrayPtr(const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type);
#endif
template _DECX_API_ de::TensorArray<uchar>*        de::CreateTensorArrayPtr(const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type);


template <typename T>
T& decx::_TensorArray<T>::index(const int x, const int y, const int z, const int tensor_id)
{
    return this->TensptrArr.ptr[tensor_id][(size_t)x * this->dp_x_wp + (size_t)y * (size_t)this->dpitch + (size_t)z];
}


template <typename T>
de::TensorArray<T>& decx::_TensorArray<T>::operator=(de::TensorArray<T>& src)
{
    decx::_TensorArray<T>& ref_src = dynamic_cast<decx::_TensorArray<T>&>(src);

    this->_attribute_assign(ref_src.width, ref_src.height, ref_src.depth, ref_src.tensor_num, ref_src._store_type);

    switch (ref_src._store_type)
    {
    

#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_malloc_same_place(&this->TensArr);
        break;
#endif

    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_malloc_same_place(&this->TensArr);
        break;

    default:
        break;
    }

    return *this;
}


template <typename T>
void decx::_TensorArray<T>::release()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_dealloc(&this->TensArr);
        break;

#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_dealloc(&this->TensArr);
        break;
#endif

    default:
        break;
    }

    decx::alloc::_host_virtual_page_dealloc(&this->TensptrArr);
}


#endif        // #ifndef _DECX_COMBINED_

#endif        // #ifndef _TENSORARRAY_H_
