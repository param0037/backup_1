#pragma once

#include "DLL_export.h"
#include "../../cv/cv_classes/cv_classes.h"

namespace de
{
    namespace vis
    {

        _DECX_API_ void ShowImage(de::vis::Img& src, const wchar_t* window_name);


        _DECX_API_ void Wait();
    }
}




void de::vis::ShowImage(de::vis::Img& src, const wchar_t* window_name)
{
    decx::_Img& _src = dynamic_cast<decx::_Img&>(src);

    decx::_ShowImage(_src.Mat.ptr, window_name, _src.width, _src.pitch, _src.height, _src.channel);
}


void de::vis::Wait()
{
    decx::_Wait();
}