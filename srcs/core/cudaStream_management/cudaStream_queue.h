/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _CUDASTREAM_QUEUE_CUH_
#define _CUDASTREAM_QUEUE_CUH_

//#ifdef _DECX_CUDA_CODES_

#include "../basic.h"
#include "cudaStream_package.h"
#include "../allocators.h"

#define _CS_STREAM_Q_INIT_SIZE_ 10


namespace decx
{
    class cudaStream_Queue;
}


class decx::cudaStream_Queue
{
private:
    size_t true_capacity;

    decx::PtrInfo<decx::cuda_stream> _cuda_stream_arr;


    bool _find_idle_stream(uint *res_dex, const int flag);


    decx::cuda_stream* add_stream_physical(const int flag);

public:
    size_t _cuda_stream_num;


    cudaStream_Queue();


    decx::cuda_stream* stream_accessor_ptr(const int flag);


    decx::cuda_stream& stream_accessor_ref(const int flag);


    void release();


    ~cudaStream_Queue() {}
};



decx::cudaStream_Queue::cudaStream_Queue()
{
    this->_cuda_stream_num = 0;
    this->true_capacity = _CS_STREAM_Q_INIT_SIZE_;

    // allocate host memory (page-locked) for decx::cuda_stream
    if (decx::alloc::_host_virtual_page_malloc(
        &this->_cuda_stream_arr, _CS_STREAM_Q_INIT_SIZE_ * sizeof(decx::cuda_stream))) {
        Print_Error_Message(4, "Failed to allocate space for cudaStream on host\n");
        exit(-1);
    }
}


decx::cuda_stream* decx::cudaStream_Queue::add_stream_physical(const int flag)
{
    if (this->_cuda_stream_num > this->true_capacity - 1) {
        // assign a temporary pointer
        decx::PtrInfo<decx::cuda_stream> tmp_ptr;
        // physically alloc space for new area
        if (decx::alloc::_host_virtual_page_malloc(&tmp_ptr, 
            (this->true_capacity + _CS_STREAM_Q_INIT_SIZE_) * sizeof(decx::cuda_stream))) {
            Print_Error_Message(4, "Failed to allocate space for cudaStream on host\n");
            exit(-1);
        }
        // copy the old data from this to temp
        memcpy(tmp_ptr.ptr, this->_cuda_stream_arr.ptr, this->true_capacity * sizeof(decx::cuda_stream));
        // refresh this->true_capacity
        this->true_capacity += _CS_STREAM_Q_INIT_SIZE_;
        // deallocate the old memory space
        decx::alloc::_host_virtual_page_dealloc(&this->_cuda_stream_arr);
        // assign the new one to the class
        this->_cuda_stream_arr = tmp_ptr;

        // alloc one from back (push_back())
        new(this->_cuda_stream_arr.ptr + this->_cuda_stream_num) decx::cuda_stream(flag);
        // increament on this->_cuda_stream_num
        ++this->_cuda_stream_num;
    }
    else {
        // alloc one from back (push_back())
        new(this->_cuda_stream_arr.ptr + this->_cuda_stream_num) decx::cuda_stream(flag);
        // increament on this->_cuda_stream_num
        ++this->_cuda_stream_num;
    }

    return (this->_cuda_stream_arr.ptr + this->_cuda_stream_num - 1);
}


bool decx::cudaStream_Queue::_find_idle_stream(uint *res_dex, const int flag)
{
    for (int i = 0; i < this->_cuda_stream_num; ++i) {
        decx::cuda_stream* _tmpS = this->_cuda_stream_arr.ptr + i;
        if (!_tmpS->_is_occupied && _tmpS->_stream_flag == flag) {
            *res_dex = i;
            _tmpS->attach();
            return true;
        }
    }
    return false;
}


decx::cuda_stream* decx::cudaStream_Queue::stream_accessor_ptr(const int flag)
{
    uint dex = 0;
    decx::cuda_stream* res_ptr = NULL;
    if (this->_find_idle_stream(&dex, flag)) {        // found an idle stream
        res_ptr = this->_cuda_stream_arr.ptr + dex;
    }
    else {          // all the streams are occupied
        res_ptr = this->add_stream_physical(flag);
    }
    return res_ptr;
}


decx::cuda_stream& decx::cudaStream_Queue::stream_accessor_ref(const int flag)
{
    uint dex = 0;
    decx::cuda_stream* res_ptr = NULL;
    if (this->_find_idle_stream(&dex, flag)) {        // found an idle stream
        res_ptr = this->_cuda_stream_arr.ptr + dex;
    }
    else {          // all the streams are occupied
        res_ptr = this->add_stream_physical(flag);
    }
    return *res_ptr;
}



void decx::cudaStream_Queue::release()
{
    // call cudaStreamDestroy on each stream
    for (int i = 0; i < this->_cuda_stream_num; ++i) {
        (this->_cuda_stream_arr.ptr + i)->release();
    }
    // deallocte the stream array
    decx::alloc::_host_virtual_page_dealloc(&this->_cuda_stream_arr);
}


namespace decx
{
    decx::cudaStream_Queue CStream;
}


//#endif

#endif
