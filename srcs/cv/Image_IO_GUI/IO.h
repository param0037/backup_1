#pragma once


#include "DLL_export.h"
#include "../../cv/cv_classes/cv_classes.h"



namespace de
{
    namespace vis
    {
        _DECX_API_
        de::DH ReadImage(const wchar_t *file_path, de::vis::Img& src);
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