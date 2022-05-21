#pragma once
#pragma comment(lib, "../bin/x64/Extra_CUDA_host.lib")

#include <windows.h>
#include <gdiplus.h>
#include <graphics.h>
#include <iostream>
#include <thread>
#include <string>

#include "../../../Image_IO_GUI/Extra_CUDA_host.h"
#include "../../DECX_world/srcs/handles/decx_handles.h"

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

// another pre-compiled DLL
static
void LoadPixels_4(::Bitmap* __bmp, uchar* __src)
{
	const uint height = __bmp->GetHeight();
	const uint width = __bmp->GetWidth();

#if 0
	size_t lin_dex = 0;
	register int RGBA;		// ARGB
	int* ptr = (int*)__src;

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			__bmp->GetPixel(j, i, (Gdiplus::Color*)(&RGBA));
			register int tmp = (RGBA & 0x00ffffff) << 8;
			tmp |= (RGBA & 0xff000000) >> 24;
			ptr[lin_dex] = *((int*)&tmp);

			++lin_dex;
		}
	}
#endif
	
	BitmapData bmpdata;
	bmpdata.Width = width;
	bmpdata.Height = height;
	bmpdata.Stride = width * 4;
	bmpdata.Scan0 = __src;
	bmpdata.Reserved = NULL;
	bmpdata.PixelFormat = PixelFormat32bppARGB;

	Rect r(0, 0, width, height);
	__bmp->LockBits(&r, ImageLockModeRead | ImageLockModeUserInputBuf,
		bmpdata.PixelFormat,
		&bmpdata);

	__bmp->UnlockBits(&bmpdata);
}


// 两个 DLL 之间用 uchar* 链接
// another pre-compiled DLL
static
void TranslateMyCls_4(uchar* __src, ::Bitmap* __bmp)
{
	const uint height = __bmp->GetHeight();
	const uint width = __bmp->GetWidth();
#if 0
	size_t lin_dex = 0;
	register int RGBA;		// ARGB
	int* ptr = (int*)__src;

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			*((int*)&RGBA) = ptr[lin_dex];
			register int tmp = (RGBA & 0xffffff00) >> 8;
			tmp |= (RGBA & 0x000000ff) << 24;
			__bmp->SetPixel(j, i, *((Gdiplus::Color*)(&tmp)));

			++lin_dex;
		}
	}
#endif

	BitmapData bmpdata;
	bmpdata.Width = width;
	bmpdata.Height = height;
	bmpdata.Stride = width * 4;
	bmpdata.Scan0 = __src;
	bmpdata.Reserved = NULL;
	bmpdata.PixelFormat = PixelFormat32bppARGB;

	Rect r(0, 0, width, height);
	__bmp->LockBits(&r, ImageLockModeWrite | ImageLockModeUserInputBuf,
		bmpdata.PixelFormat,
		&bmpdata);

	__bmp->UnlockBits(&bmpdata);
}


__declspec(align(4)) struct uchar4
{
	uchar x, y, z, w;
};


struct uchar3
{
	uchar x, y, z;
};


static
void TranslateMyCls_1(uchar* __src, ::Bitmap* __bmp)
{
	const uint height = __bmp->GetHeight();
	const uint width = __bmp->GetWidth();
#if 1
	BitmapData bmpdata;
	
	Rect r(0, 0, width, height);

	__bmp->LockBits(&r, ImageLockModeWrite,
		PixelFormat32bppARGB, &bmpdata);

	uchar4* ptr = reinterpret_cast<uchar4*>(bmpdata.Scan0);
	const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);

	register uchar __tmp = 0;
	register uchar4* _tmp_ptr = NULL;
	
	for (size_t i = 0; i < total; ++i) {
		__tmp = __src[i];
		_tmp_ptr = &ptr[i];
		_tmp_ptr->x = __tmp;
		_tmp_ptr->y = __tmp;
		_tmp_ptr->z = __tmp;
		_tmp_ptr->w = 255;
	}
	

	__bmp->UnlockBits(&bmpdata);
#endif
#if 0
	BitmapData bmpdata;
	bmpdata.Width = width;
	bmpdata.Height = height;
	bmpdata.Stride = width * 4;
	bmpdata.Scan0 = __src;
	bmpdata.Reserved = NULL;
	bmpdata.PixelFormat = PixelFormat16bppGrayScale;

	Rect r(0, 0, width, height);
	__bmp->LockBits(&r, ImageLockModeWrite | ImageLockModeUserInputBuf,
		bmpdata.PixelFormat,
		&bmpdata);

	__bmp->UnlockBits(&bmpdata);
#endif
}




// This DLL, 传入一个 uchar 数组的地址（已 malloc）
__declspec(dllexport)
const char* _ReadImage(const wchar_t* file_path, uchar** return_Img, const int flag, uint *W, uint *H, uint *channel, int *Error_type)
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

	*W = width;
	*H = height;
	
	if (flag == DE_UC1)		*channel = 1;
	if (flag == DE_UC3)		*channel = 3;
	if (flag == DE_UC4 || flag == DE_IMG_4_ALIGN)	*channel = 4;

	__cudahostalloc(reinterpret_cast<void**>(return_Img), width * height * (*channel), cudaHostAllocDefault);
	
	if (flag == DE_UC4 || flag == DE_IMG_4_ALIGN)
	{
		LoadPixels_4(lpbmp, *return_Img);
	}
	//关闭GDI+   
	GdiplusShutdown(ulToken);

	*Error_type = decx::DECX_SUCCESS;
	return "No error";
}



static
int GetEncoderClsid(const WCHAR* format, CLSID* pClsid)
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





//void Save(const wchar_t* file_path,
//	uchar* __src,
//	const uint height,
//	const uint width,
//	const uint channel,
//	const int save_flag)
//{
//	Bitmap* lpbmp = new ::Bitmap(width, height);
//	if (channel == 1) {
//		TranslateMyCls_1(__src, lpbmp);
//	}
//	if (channel == 4) {
//		TranslateMyCls_4(__src, lpbmp);
//	}
//	CLSID imgClsid;
//	switch (save_flag)
//	{
//	case ImgSave_jpg:
//		GetEncoderClsid(L"image/jpeg", &imgClsid);
//		break;
//	case ImgSave_png:
//		GetEncoderClsid(L"image/png", &imgClsid);
//		break;
//
//	default:
//		break;
//	}
//	lpbmp->Save(file_path, imgClsid, NULL);
//}