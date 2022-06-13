#pragma once



#define cudaHostAllocDefault                0x00  /**< Default page-locked allocation flag */
#define cudaHostAllocPortable               0x01  /**< Pinned memory accessible by all CUDA contexts */
#define cudaHostAllocMapped                 0x02  /**< Map allocation into device space */
#define cudaHostAllocWriteCombined          0x04  /**< Write-combined memory */


__declspec(dllexport)
void __cudahostalloc(void** src, const size_t bytes, const int flag);



__declspec(dllexport)
void __cudamalloc(void** src, const size_t bytes);