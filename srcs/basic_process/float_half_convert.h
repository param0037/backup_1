/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once


#include "../classes/core_types.h"



namespace de
{

    _DECX_API_ de::Half Float2Half(const float& __x);


    _DECX_API_ float Half2Float(const de::Half& __x);
}



using namespace decx;
using namespace ::alloc;


de::Half de::Float2Half(const float& __x)
{
    __half tmp = __float2half(__x);
    return *((de::Half*)&tmp);
}


float de::Half2Float(const de::Half& __x)
{
    float tmp = __half2float(*((__half*)&__x));
    return tmp;
}
