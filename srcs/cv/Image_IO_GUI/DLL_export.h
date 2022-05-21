#pragma once


#include "../../cv/cv_classes/cv_classes.h"

#ifdef Windows
#pragma comment(lib, "../../../../bin/x64/Image_IO_GUI.lib")
#endif


namespace decx
{
    _DECX_API_ const char* _ReadImage(const wchar_t* file_path, decx::PtrInfo<uchar>* return_Img, uint* W, uint* pitch, uint* H, uint* channel, int* Error_type);



    _DECX_API_ void _ShowImage(uchar* src, const wchar_t* window_name, int width, int pitch, int height, int channel);



    _DECX_API_ void _Wait();


    _DECX_API_ void GDIPlusInit();
}