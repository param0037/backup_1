#pragma once


#include "../memory_management/MemoryPool_Hv.h"
#include "../memory_management/MemoryPool_Hf.h"
#include "../memory_management/MemoryPool_D.h"



namespace decx 
{

    decx::MemPool_Hv mem_pool_Hv;

    decx::MemPool_Hf mem_pool_Hf;

    decx::MemPool_D mem_pool_D;
}


namespace de
{
    _DECX_API_ void release_all_tmp();
}



void de::release_all_tmp()
{

    decx::mem_pool_Hv.release();

    decx::mem_pool_Hf.release();

    decx::mem_pool_D.release();

}