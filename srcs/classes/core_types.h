/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#pragma once


#include "vector.h"
#include "matrix.h"
#include "tensor.h"
#include "MatrixArray.h"
#include "classes_util.h"


enum DECX_THREADS_FLAGS
{
    MULTI_HOST_THREADS = 0,
    SINGLE_HOST_THREADS = 1
};


enum DECX_DEV_TENSOR_STORE_TYPE
{
    Tensor_YXZ = 0x00,
    Tensor_ZYX = 0x01
};




enum Tensor_Store_Types
{
    channel_separate = 0,
    channel_adjacent = 1
};