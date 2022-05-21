/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#include "include.h"
#include "../handles/decx_handles.h"



#ifdef Windows
#define SetConsoleColor(_color_flag) SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), _color_flag)

#define ResetConsoleColor SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 7)
#endif



#ifdef _DECX_CUDA_CODES_
static inline const char* _cudaGetErrorEnum(cudaError_t error) noexcept
{
    return cudaGetErrorName(error);
}


template <typename T>
void check(T result, char const* const func, const char* const file, int const line) 
{
    if (result) {
        SetConsoleColor(4);
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        ResetConsoleColor;
        exit(EXIT_FAILURE);
    }
}

//#ifdef Windows
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#endif



#define Error_Handle(__handle, _operator, _return)    \
{    \
if (__handle _operator error_type != decx::DECX_SUCCESS) {    \
    return _return;    \
}    \
}


// --------------------------- ERROR_STATEMENTS -------------------------------

#define NOT_INIT                                "CUDA should be initialized first\n"
#define SUCCESS                                    "No error\n"
#define FFT_ERROR_LENGTH                        "Each dim should be able to be separated by 2, 3 and 5\n"
#define FFT_ERROR_WIDTH                            "Width should be able to be separated by 2, 3 and 5\n"
#define FFT_ERROR_HEIGHT                        "Height should be able to be separated by 2, 3 and 5\n"
#define ALLOC_FAIL                                "Fail to allocate memory\n"
#define DIM_NOT_EQUAL                            "Dim(s) is(are) not equal to each other\n"
#define MEANINGLESS_FLAG                        "This flag is meaningless in current context\n"



#define Print_Error_Message(_color, _statement)    \
{    \
SetConsoleColor(_color);    \
printf(_statement);    \
ResetConsoleColor;    \
}



namespace decx
{
#ifndef GNU_CPUcodes
    static void Not_init(de::DH* handle) noexcept
    {
        handle->error_string = (char*)NOT_INIT;
        handle->error_type = decx::DECX_FAIL_not_init;
    }
#endif


    static void Success(de::DH* handle) noexcept
    {
        handle->error_string = (char*)SUCCESS;
        handle->error_type = decx::DECX_SUCCESS;
    }



    static void MDim_Not_Matching(de::DH* handle) noexcept
    {
        handle->error_string = (char*)"Matrix Dims don't match each other";
        handle->error_type = decx::DECX_FAIL_DimsNotMatching;
    }


    static void Matrix_number_not_matching(de::DH* handle) noexcept
    {
        handle->error_string = (char*)"The number of matrices don't match each other";
        handle->error_type = decx::DECX_FAIL_MNumNotMatching;
    }



    static void GEMM_DimNMatch(de::DH* handle) noexcept
    {
        handle->error_type = decx::DECX_FAIL_DimsNotMatching;
        handle->error_string = (char*)"The width of matrix A and the height of matrix B are required to be same";
    }



    static void TDim_Not_Matching(de::DH* handle) noexcept
    {
        handle->error_string = (char*)"Tensor Dims don't match each other";
        handle->error_type = decx::DECX_FAIL_DimsNotMatching;
    }


    static void StoreFormatError(de::DH* handle) noexcept
    {
        handle->error_string = (char*)"The store type is not suitable";
        handle->error_type = decx::DECX_FAIL_StoreError;
    }



    static void MeaninglessFlag(de::DH* handle) noexcept
    {
        handle->error_string = (char*)"This flag is meaningless";
        handle->error_type = decx::DECX_FAIL_ErrorFlag;
    }



    namespace err
    {
        static void FFT_Error_length(de::DH* handle)    noexcept
        {
            handle->error_string = (char*)FFT_ERROR_LENGTH;
            handle->error_type = decx::DECX_FAIL_FFT_error_length;
        }


        static void AllocateFailure(de::DH* handle)    noexcept
        {
            handle->error_string = (char*)ALLOC_FAIL;
            handle->error_type = decx::DECX_FAIL_ALLOCATION;
        }
    }
}