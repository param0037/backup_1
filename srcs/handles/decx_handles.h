#pragma once


namespace decx
{
    enum DECX_error_types
    {
        DECX_SUCCESS                = 0x00,

        DECX_FAIL_not_init            = 0x01,

        DECX_FAIL_FFT_error_length    = 0x02,

        DECX_FAIL_DimsNotMatching    = 0x03,
        DECX_FAIL_Complex_comparing = 0x04,

        DECX_FAIL_ConvBadKernel        = 0x05,
        DECX_FAIL_StepError            = 0x06,

        DECX_FAIL_ChannelError        = 0x07,
        DECX_FAIL_CVNLM_BadArea        = 0x08,

        DECX_FAIL_FileNotExist        = 0x09,

        DECX_GEMM_DimsError            = 0x0a,

        DECX_FAIL_ErrorFlag            = 0x0b,

        DECX_FAIL_DimError            = 0x0c,

        DECX_FAIL_ErrorParams        = 0x0d,

        DECX_FAIL_StoreError        = 0x0e,

        DECX_FAIL_MNumNotMatching    = 0x0f,

        DECX_FAIL_ALLOCATION        = 0x10
    };
}


namespace de
{
    typedef struct DECX_Handle
    {
        int error_type;
        char* error_string;
    }DH;
}



//enum ImageSaveTypes
//{
//    ImgSave_jpg = 0,
//    ImgSave_png = 1,
//    ImgSave_bmp = 2,
//    ImgSave_gif = 3
//};