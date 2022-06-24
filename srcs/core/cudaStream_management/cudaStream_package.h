/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _CUDASTREAM_PACKAGE_CUH_
#define _CUDASTREAM_PACKAGE_CUH_

//#ifdef _DECX_CUDA_CODES_

#include "../basic.h"

namespace decx
{
    class cuda_stream;
}


class decx::cuda_stream
{
private:
    cudaStream_t _S;

public:
    int _stream_flag;
    bool _is_occupied;

    cuda_stream(const int flag);


    void detach();


    void attach();

    /* Call cudaStreamSynchronize() and the parameter is this->_S */
    void this_stream_sync();

    /* Return a referance of cudaStream_t object */
    cudaStream_t& get_raw_stream_ref();

    /* Return a pointer of cudaStream_t object */
    cudaStream_t* get_raw_stream_ptr();


    void release();


    ~cuda_stream() {}
};


decx::cuda_stream::cuda_stream(const int flag)
{
    checkCudaErrors(cudaStreamCreateWithFlags(&this->_S, flag));
    this->_stream_flag = flag;
}


void decx::cuda_stream::detach()
{
    this->_is_occupied = false;
}


void decx::cuda_stream::attach()
{
    this->_is_occupied = true;
}


void decx::cuda_stream::this_stream_sync()
{
    checkCudaErrors(cudaStreamSynchronize(this->_S));
}


cudaStream_t& decx::cuda_stream::get_raw_stream_ref()
{
    return this->_S;
}


cudaStream_t* decx::cuda_stream::get_raw_stream_ptr()
{
    return &(this->_S);
}


void decx::cuda_stream::release()
{
    checkCudaErrors(cudaStreamDestroy(this->_S));
}


//#endif

#endif