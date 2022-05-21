/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

// addition
#ifdef _DECX_CUDA_CODES_
#include "matrix/cuda_add.h"
#include "Vector/cuda_add.h"

// subtraction
#include "matrix/cuda_subtract.h"
#include "Vector/cuda_subtract.h"

// multiply
//#include "matrix/cuda_multiply.h"
#include "Vector/cuda_multiply.h"

////// divide
#include "matrix/cuda_divide.h"
#include "Vector/cuda_divide.h"

#include "matrix/cuda_fma.h"
#include "matrix/cuda_fms.h"
#include "Vector/cuda_fms.h"

#endif

#ifdef _DECX_CPU_CODES_

#endif
