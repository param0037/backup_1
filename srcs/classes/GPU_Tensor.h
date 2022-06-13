/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _GPU_TENSOR_H_
#define _GPU_TENSOR_H_

#include "../core/basic.h"
#include "Tensor.h"


namespace de
{
    /** 用 channel_adjcent 的方式存储，将 channel 的数量凑成8倍数, CPU端可以用__m256，GPU端可以用float2
    * 将两字节的 half 类型凑成四倍数,GPU端可以凑成偶倍的 half2, 将 double 等八字节的数据类型凑成偶数*/
    template <typename T>
    class _DECX_API_ GPU_Tensor
    {
    public:
        GPU_Tensor() {}


        virtual uint Width() = 0;


        virtual uint Height() = 0;


        virtual uint Depth() = 0;


        virtual de::GPU_Tensor<T>& operator=(de::GPU_Tensor<T>& src) = 0;


        virtual void Load_from_host(de::Tensor<T>& src) = 0;


        virtual void Load_to_host(de::Tensor<T>& dst) = 0;


        virtual void release() = 0;
    };
}


#ifndef _DECX_COMBINED_

/**
* The data storage structure is shown below
*
*            <-------------------- dp_x_wp(4x) ------------------>
*            <--------------- width ---------------->
*             <-dpitch->
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] T
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] |
*            [[x x x...] [x x x...] ... ...[x x x...] [0 0 0 0]] |
*            ...                                     ...   ...   |    height
*            ...                                     ...   ...   |
*            ...                                     ...   ...   |
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
    class _GPU_Tensor : public de::GPU_Tensor<T>
    {
    private:
        void _attribute_assign(const uint _width, const uint _height, const uint _depth);


        void alloc_data_space();


        void re_alloc_data_space();

    public:
        uint width,
            height,
            depth;

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


        void construct(const uint _width, const uint _height, const uint _depth);


        void re_construct(const uint _width, const uint _height, const uint _depth);


        _GPU_Tensor();


        _GPU_Tensor(const uint _width, const uint _height, const uint _depth);



        virtual uint Width() { return this->width; }


        virtual uint Height() { return this->height; }


        virtual uint Depth() { return this->depth; }


        virtual de::GPU_Tensor<T>& operator=(de::GPU_Tensor<T>& src);


        virtual void Load_from_host(de::Tensor<T>& src);


        virtual void Load_to_host(de::Tensor<T>& dst);


        virtual void release();
    };
}



void decx::_GPU_Tensor<float>::_attribute_assign(const uint _width, const uint _height, const uint _depth)
{
    this->width = _width;
    this->height = _height;
    this->depth = _depth;

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



void decx::_GPU_Tensor<int>::_attribute_assign(const uint _width, const uint _height, const uint _depth)
{
    this->width = _width;
    this->height = _height;
    this->depth = _depth;

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
void decx::_GPU_Tensor<de::Half>::_attribute_assign(const uint _width, const uint _height, const uint _depth)
{
    this->width = _width;
    this->height = _height;
    this->depth = _depth;

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


void decx::_GPU_Tensor<double>::_attribute_assign(const uint _width, const uint _height, const uint _depth)
{
    this->width = _width;
    this->height = _height;
    this->depth = _depth;

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



void decx::_GPU_Tensor<uchar>::_attribute_assign(const uint _width, const uint _height, const uint _depth)
{
    this->width = _width;
    this->height = _height;
    this->depth = _depth;

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
void decx::_GPU_Tensor<T>::alloc_data_space()
{
    if (decx::alloc::_device_malloc(&this->Tens, this->total_bytes)) {
        Print_Error_Message(4, "Tensor malloc failed! Please check if there is enough space in your device.");
        exit(-1);
    }
    checkCudaErrors(cudaMemset(this->Tens.ptr, 0, this->total_bytes));
}



template <typename T>
void decx::_GPU_Tensor<T>::re_alloc_data_space()
{
    if (decx::alloc::_device_realloc(&this->Tens, this->total_bytes)) {
        Print_Error_Message(4, "Tensor malloc failed! Please check if there is enough space in your device.");
        exit(-1);
    }
    checkCudaErrors(cudaMemset(this->Tens.ptr, 0, this->total_bytes));
}



template<typename T>
void decx::_GPU_Tensor<T>::construct(const uint _width, const uint _height, const uint _depth)
{
    this->_attribute_assign(_width, _height, _depth);

    this->alloc_data_space();
}



template<typename T>
void decx::_GPU_Tensor<T>::re_construct(const uint _width, const uint _height, const uint _depth)
{
    this->_attribute_assign(_width, _height, _depth);

    this->re_alloc_data_space();
}



template<typename T>
decx::_GPU_Tensor<T>::_GPU_Tensor(const uint _width, const uint _height, const uint _depth)
{
    this->_attribute_assign(_width, _height, _depth);

    this->alloc_data_space();
}



template<typename T>
decx::_GPU_Tensor<T>::_GPU_Tensor()
{
    this->_attribute_assign(0, 0, 0);
}



template <typename T>
de::GPU_Tensor<T>& decx::_GPU_Tensor<T>::operator=(de::GPU_Tensor<T>& src)
{
    decx::_GPU_Tensor<T>& ref_src = dynamic_cast<decx::_GPU_Tensor<T> &>(src);

    this->Tens.block = ref_src.Tens.block;

    this->_attribute_assign(ref_src.width, ref_src.height, ref_src.depth);

    decx::alloc::_device_malloc_same_place(&this->Tens);

    return *this;
}



template <typename T>
void decx::_GPU_Tensor<T>::Load_from_host(de::Tensor<T>& src)
{
    decx::_Tensor<T>& ref_src = dynamic_cast<decx::_Tensor<T>&>(src);

    checkCudaErrors(cudaMemcpy(this->Tens.ptr, ref_src.Tens.ptr, this->total_bytes, cudaMemcpyHostToDevice));
}



template <typename T>
void decx::_GPU_Tensor<T>::Load_to_host(de::Tensor<T>& dst)
{
    decx::_Tensor<T>& ref_dst = dynamic_cast<decx::_Tensor<T>&>(dst);

    checkCudaErrors(cudaMemcpy(ref_dst.Tens.ptr, this->Tens.ptr, this->total_bytes, cudaMemcpyDeviceToHost));
}



template <typename T>
void decx::_GPU_Tensor<T>::release()
{
    decx::alloc::_device_dealloc(&this->Tens);
}


namespace de
{
    template <typename T>
    de::GPU_Tensor<T>* CreateGPUTensorPtr();


    template <typename T>
    de::GPU_Tensor<T>& CreateGPUTensorRef();


    template <typename T>
    de::GPU_Tensor<T>* CreateGPUTensorPtr(const uint _width, const uint _height, const uint _depth);


    template <typename T>
    de::GPU_Tensor<T>& CreateGPUTensorRef(const uint _width, const uint _height, const uint _depth);
}



template <typename T>
de::GPU_Tensor<T>& de::CreateGPUTensorRef()
{
    return *(new decx::_GPU_Tensor<T>());
}


template <typename T>
de::GPU_Tensor<T>* de::CreateGPUTensorPtr()
{
    return new decx::_GPU_Tensor<T>();
}

template _DECX_API_ de::GPU_Tensor<int>& de::CreateGPUTensorRef();

template _DECX_API_ de::GPU_Tensor<float>& de::CreateGPUTensorRef();

template _DECX_API_ de::GPU_Tensor<double>& de::CreateGPUTensorRef();

template _DECX_API_ de::GPU_Tensor<uchar>& de::CreateGPUTensorRef();

#ifdef _DECX_CUDA_CODES_
template _DECX_API_ de::GPU_Tensor<de::Half>& de::CreateGPUTensorRef();
#endif



template _DECX_API_ de::GPU_Tensor<int>* de::CreateGPUTensorPtr();

template _DECX_API_ de::GPU_Tensor<float>* de::CreateGPUTensorPtr();

template _DECX_API_ de::GPU_Tensor<double>* de::CreateGPUTensorPtr();

template _DECX_API_ de::GPU_Tensor<uchar>* de::CreateGPUTensorPtr();

#ifdef _DECX_CUDA_CODES_
template _DECX_API_ de::GPU_Tensor<de::Half>* de::CreateGPUTensorPtr();
#endif



template <typename T>
de::GPU_Tensor<T>& de::CreateGPUTensorRef(const uint _width, const uint _height, const uint _depth)
{
    return *(new decx::_GPU_Tensor<T>(_width, _height, _depth));
}


template <typename T>
de::GPU_Tensor<T>* de::CreateGPUTensorPtr(const uint _width, const uint _height, const uint _depth)
{
    return new decx::_GPU_Tensor<T>(_width, _height, _depth);
}

template _DECX_API_ de::GPU_Tensor<int>& de::CreateGPUTensorRef(const uint _width, const uint _height, const uint _depth);

template _DECX_API_ de::GPU_Tensor<float>& de::CreateGPUTensorRef(const uint _width, const uint _height, const uint _depth);

template _DECX_API_ de::GPU_Tensor<double>& de::CreateGPUTensorRef(const uint _width, const uint _height, const uint _depth);

template _DECX_API_ de::GPU_Tensor<uchar>& de::CreateGPUTensorRef(const uint _width, const uint _height, const uint _depth);

#ifdef _DECX_CUDA_CODES_
template _DECX_API_ de::GPU_Tensor<de::Half>& de::CreateGPUTensorRef(const uint _width, const uint _height, const uint _depth);
#endif



template _DECX_API_ de::GPU_Tensor<int>* de::CreateGPUTensorPtr(const uint _width, const uint _height, const uint _depth);

template _DECX_API_ de::GPU_Tensor<float>* de::CreateGPUTensorPtr(const uint _width, const uint _height, const uint _depth);

template _DECX_API_ de::GPU_Tensor<double>* de::CreateGPUTensorPtr(const uint _width, const uint _height, const uint _depth);

template _DECX_API_ de::GPU_Tensor<uchar>* de::CreateGPUTensorPtr(const uint _width, const uint _height, const uint _depth);

#ifdef _DECX_CUDA_CODES_
template _DECX_API_ de::GPU_Tensor<de::Half>* de::CreateGPUTensorPtr(const uint _width, const uint _height, const uint _depth);
#endif


#endif      // #ifndef _DECX_COMBINED_


#endif      // #ifndef _GPU_TENSOR_H_