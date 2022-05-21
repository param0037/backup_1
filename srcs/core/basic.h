/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#ifndef _BASIC_H_
#define _BASIC_H_

#include "configuration.h"
#include "defines.h"
#include "error.h"
#include "../handles/decx_handles.h"
#include "vector_defines.h"



#ifdef _DECX_CUDA_CODES_
#define CONSTANT_MEM_SIZE 65536
__constant__ uchar Const_Mem[CONSTANT_MEM_SIZE];
#endif


enum ProcTypes
{
    SyncDataHostToDevice = 0,
    SyncDataDeviceToHost = 1
};

enum dev_cls_create_flags
{
    dev_cls_CopyHost = 0,
    dev_cls_UseHost = 1
};


template <typename T>
struct T2
{
    T x, y;
};


#ifndef GNU_CPUcodes
#define PitchDefault     0x0000
#define PitchOdd         0x0001
#define PitchAuto         0x0002
#endif

#endif