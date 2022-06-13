/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*
*    cvt_colors uses CPU and avx instead of CUDA, since it does not need much
* calculation. This module call the APIs from DECX_cpu.dll(.so)
*/

#ifndef _CVT_COLORS_DEF_H_
#define _CVT_COLORS_DEF_H_

#include "../cv_classes/cv_classes.h"

#ifdef Windows
#ifdef _DECX_COMBINED_
#pragma comment(lib, "../../../../bin/x64/DECX_cpu.lib")
#endif
#endif

namespace decx
{
    /*
    * dims.x : It is the true pitch of pixel data matrix, scale of uchar
    * dims.y : It is the height of pixel data matrix
    */
    _DECX_API_ void _BGR2Gray_UC2UC_caller(float* src, float* dst, const int2 dims);

    /*
    * dims.x : It is the true pitch of pixel data matrix, scale of uchar
    * dims.y : It is the height of pixel data matrix
    */
    _DECX_API_ void _Preserve_B_UC2UC_caller(float* src, float* dst, const int2 dims);

    /*
    * dims.x : It is the true pitch of pixel data matrix, scale of uchar
    * dims.y : It is the height of pixel data matrix
    */
    _DECX_API_ void _Preserve_G_UC2UC_caller(float* src, float* dst, const int2 dims);

    /*
    * dims.x : It is the true pitch of pixel data matrix, scale of uchar
    * dims.y : It is the height of pixel data matrix
    */
    _DECX_API_ void _Preserve_R_UC2UC_caller(float* src, float* dst, const int2 dims);
}


#endif