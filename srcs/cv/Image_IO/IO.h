#pragma once
#pragma comment(lib, "../bin/x64/Extra_CUDA_host.lib")

#include <windows.h>
#include <gdiplus.h>
#include <graphics.h>
#include <iostream>
#include <thread>
#include <string>

#include "../../../Image_IO_GUI/Extra_CUDA_host.h"
#include "../../core/allocators.h"
#include "../../DECX_world/srcs/handles/decx_handles.h"
#include "../../core/utils/decx_utils_functions.h"
#include "../../cv/cv_classes/cv_classes.h"
#include "../../cv/cv_classes/cv_cls_MFuncs.h"



enum ImgConstructType
{
    DE_UC1 = 1,
    DE_UC3 = 3,
    DE_UC4 = 4,
    DE_IMG_DEFAULT = 5,
    DE_IMG_4_ALIGN = 6
};


using namespace Gdiplus;


typedef unsigned int uint;
typedef unsigned char uchar;


#pragma comment(lib,"D:/Windows Kits/10/Lib/10.0.18362.0/um/x64/gdiplus.lib")  //导入GDI+库


namespace decx
{
    static void LoadPixels_4(::Bitmap* __bmp, uchar* __src, int pitch);

    static void TranslateMyCls_4(uchar* __src, ::Bitmap* __bmp, const int pitch);

    static void TranslateMyCls_1(uchar* __src, ::Bitmap* __bmp, const int pitch);


    //__declspec(dllexport) 
    const char* _ReadImage(const wchar_t* file_path, decx::PtrInfo<uchar>* return_Img, uint* W, uint* pitch, uint* H, uint* channel, int* Error_type);


    static int GetEncoderClsid(const WCHAR* format, CLSID* pClsid);
}


// another pre-compiled DLL
static void decx::LoadPixels_4(::Bitmap* __bmp, uchar* __src, int pitch)
{
    const uint height = __bmp->GetHeight();
    const uint width = __bmp->GetWidth();

    BitmapData bmpdata;
    bmpdata.Width = width;
    bmpdata.Height = height;
    bmpdata.Stride = pitch * 4;
    bmpdata.Scan0 = __src;
    bmpdata.Reserved = NULL;
    bmpdata.PixelFormat = PixelFormat32bppARGB;

    Rect r(0, 0, width, height);
    __bmp->LockBits(&r, ImageLockModeRead | ImageLockModeUserInputBuf,
        bmpdata.PixelFormat,
        &bmpdata);

    __bmp->UnlockBits(&bmpdata);
}



static void decx::TranslateMyCls_4(uchar* __src, ::Bitmap* __bmp, const int pitch)
{
    const uint height = __bmp->GetHeight();
    const uint width = __bmp->GetWidth();

    BitmapData bmpdata;
    bmpdata.Width = width;
    bmpdata.Height = height;
    bmpdata.Stride = pitch * 4;
    bmpdata.Scan0 = __src;
    bmpdata.Reserved = NULL;
    bmpdata.PixelFormat = PixelFormat32bppARGB;

    Rect r(0, 0, width, height);
    __bmp->LockBits(&r, ImageLockModeWrite | ImageLockModeUserInputBuf,
        bmpdata.PixelFormat,
        &bmpdata);

    __bmp->UnlockBits(&bmpdata);
}



static void decx::TranslateMyCls_1(uchar* __src, ::Bitmap* __bmp, const int pitch)
{
    const int height = __bmp->GetHeight();
    const int width = __bmp->GetWidth();
    
    BitmapData bmpdata;
    
    Rect r(0, 0, width, height);

    __bmp->LockBits(&r, ImageLockModeWrite,
        PixelFormat32bppARGB, &bmpdata);

    uchar4* ptr = reinterpret_cast<uchar4*>(bmpdata.Scan0);

    register uchar __tmp = 0;
    register uchar4* _tmp_ptr = NULL;
    size_t dex = 0, dex_buffer = 0;
    const size_t _bias = (size_t)(pitch - width);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            __tmp = __src[dex];
            _tmp_ptr = &ptr[dex_buffer];
            _tmp_ptr->x = __tmp;
            _tmp_ptr->y = __tmp;
            _tmp_ptr->z = __tmp;
            _tmp_ptr->w = 255;
            ++dex_buffer;
            ++dex;
        }
        dex += _bias;
    }
    

    __bmp->UnlockBits(&bmpdata);
}



#define _IMG_ALIGN_ 16

const char* decx::_ReadImage(const wchar_t* file_path, decx::PtrInfo<uchar>* return_Img, uint *W, uint *pitch, uint *H, uint *channel, int *Error_type)
{
    //调用GDI+开始函数  
    ULONG_PTR  ulToken;
    GdiplusStartupInput    gdiplusStartupInput;
    ::GdiplusStartup(&ulToken, &gdiplusStartupInput, NULL);

    // inside DLL, destroyed when the DLL built-in function exit
    ::Bitmap* lpbmp = new ::Bitmap(file_path);

    if (lpbmp == NULL) {
        *Error_type = decx::DECX_FAIL_FileNotExist;
        return "No such file";
    }

    const uint height = lpbmp->GetHeight();
    const uint width = lpbmp->GetWidth();

    *pitch = decx::utils::ceil<uint>(width, _IMG_ALIGN_) * _IMG_ALIGN_;

    *W = width;
    *H = height;
    
    *channel = 4;

    if (return_Img->ptr == NULL) {
        if (decx::alloc::_host_fixed_page_malloc(return_Img, (size_t)*pitch * (size_t)height * (size_t)*channel)) {
            Print_Error_Message(4, ALLOC_FAIL);
            return "Fail to allocate memory";
        }
    }
    else {
        decx::alloc::_host_fixed_page_dealloc(return_Img);
        if (decx::alloc::_host_fixed_page_malloc(return_Img, (size_t)*pitch * (size_t)height * (size_t)*channel)) {
            Print_Error_Message(4, ALLOC_FAIL);
            return "Fail to allocate memory";
        }
    }
    
    decx::LoadPixels_4(lpbmp, return_Img->ptr, *pitch);
    //关闭GDI+   
    GdiplusShutdown(ulToken);

    *Error_type = decx::DECX_SUCCESS;
    return "No error";
}



static int decx::GetEncoderClsid(const WCHAR* format, CLSID* pClsid)
{
    UINT  num = 0;          // number of image encoders
    UINT  size = 0;         // size of the image encoder array in bytes

    ImageCodecInfo* pImageCodecInfo = NULL;

    GetImageEncodersSize(&num, &size);
    if (size == 0)
        return -1;  // Failure

    pImageCodecInfo = (ImageCodecInfo*)(malloc(size));
    if (pImageCodecInfo == NULL)
        return -1;  // Failure

    GetImageEncoders(num, size, pImageCodecInfo);

    for (UINT j = 0; j < num; ++j)
    {
        if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0)
        {
            *pClsid = pImageCodecInfo[j].Clsid;
            free(pImageCodecInfo);
            return j;  // Success
        }
    }

    free(pImageCodecInfo);
    return -1;  // Failure
}



namespace de
{
    namespace vis
    {
        _DECX_API_ de::DH ReadImage(const wchar_t* file_path, de::vis::Img& src);
    }
}



de::DH de::vis::ReadImage(const wchar_t* file_path, de::vis::Img& src)
{
    decx::_Img& _src = dynamic_cast<decx::_Img&>(src);

    de::DH handle;
    handle.error_string =
        const_cast<char*>(
            decx::_ReadImage(file_path, &(_src.Mat), &(_src.width), &(_src.pitch), &(_src.height), &(_src.channel), &(handle.error_type)));

    if (handle.error_type != decx::DECX_SUCCESS) {
        return handle;
    }
    _src.element_num =
        static_cast<size_t>(_src.width) * static_cast<size_t>(_src.height) * static_cast<size_t>(_src.channel);

    _src.ImgPlane = static_cast<size_t>(_src.width) * static_cast<size_t>(_src.height);
    _src.total_bytes = _src.element_num;
    _src._element_num = (size_t)_src.pitch * (size_t)_src.height;

    handle.error_type = decx::DECX_SUCCESS;
    return handle;
}