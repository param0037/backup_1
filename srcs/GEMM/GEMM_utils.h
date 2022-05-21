/**
*	---------------------------------------------------------------------
*	Author : Wayne
*   Date   : 2021.9.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.9.16
*/

#pragma once

#include "../core/basic.h"


#define GEMM_BlockDim 16

//
//static bool is_Nonintegral(uint* dim1, uint* dim2)
//{
//	return ((*dim1 % GEMM_BlockDim) || (*dim2 % GEMM_BlockDim));
//}