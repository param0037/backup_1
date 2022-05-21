/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16 
*    ---------------------------------------------------------------------
*        This Program should be compiled at C++ 11 or versions above
*    This Program supports from cuda 2.0 to cuda 10.2. However, the 
*    new features of Nvidia like RT core are not included yet. And it
*    supports only single GPU.
*      
*      ���ӳ�������� c++ 11 ������ƽ̨���Ӻͱ��롣���ӳ���֧�ֵ� CUDA �汾
*    �� 3.0 �� 10.2�����ǣ�����Ӣΰ��������ԣ��������׷�ٺ����Լ� tensor cores 
*    �ĵ��ã���δ֧�֡����⣬���ӳ���ֻ֧�ֵ� GPU ������
*/


#define STRING_MACRO(x) #x
#define STR2(x) STRING_MACRO(x)
#ifdef __CUDA_ARCH__
#pragma message("Compiling codes for CUDA device sm & compute = " STR2(__CUDA_ARCH__))
#endif



#define _decx_basic
#define _decx_basic_proc
#define _decx_basic_calc
#define _decx_ops
#define _decx_dnn
#define _decx_fft


// core of DECX
#ifdef _decx_basic
#include "../srcs/classes/matrix.h"
#include "../srcs/classes/vector.h"
#include "../srcs/classes/tensor.h"
#include "../srcs/classes/MatrixArray.h"
#include "../srcs/classes/TensorArray.h"
#include "../srcs/classes/GPU_Matrix.h"
#include "../srcs/classes/GPU_MatrixArray.h"
#include "../srcs/classes/GPU_Vector.h"
#endif
//
//

#include "../srcs/convolution/CUDA/slide_window/Conv2.h"
#include "../srcs/convolution/CUDA/slide_window/conv2_mc.h"
#include "../srcs/convolution/CUDA/slide_window/conv2_large_kernel.h"
#include "../srcs/convolution/CUDA/im2col/conv2_mk_im2col.h"

#ifdef _decx_basic_calc
// 卷积
#include "../srcs/basic_calculations/operators/operators.h"



// 通用矩阵乘
#include "../srcs/GEMM/CUDA/large_squares/GEMM.h"
#include "../srcs/GEMM/CUDA/large_squares/GEMM3.h"
#include "../srcs/GEMM/CUDA/extreme_shapes/GEMM_long_Lr.h"
#endif

#include "../srcs/basic_process/float_half_convert.h"
#include "../srcs/basic_process/channel_alteration/MatArray_channel_sum.h"
#include "../srcs/Dot product/CUDA/fp32/dot_fp32.h"
#include "../srcs/basic_process/type_cast/Matrix2Vector.h"
#include "../srcs/basic_process/type_statistics/maximum.h"
#include "../srcs/basic_process/type_statistics/minimum.h"
#include "../srcs/basic_process/type_statistics/summing.h"
#include "../srcs/basic_process/transpose/transpose.h"
#include "../srcs/basic_process/reverse/Mat_cuda_rev.cuh"


#ifdef _decx_fft
#include "../srcs/fft/CUDA/fft_utils.cuh"
#include "../srcs/fft/CUDA/1D/FFT/FFT1D.h"
#include "../srcs/fft/CUDA/1D/IFFT/IFFT1D.h"
#include "../srcs/fft/CUDA/2D/FFT/FFT2D.h"
#include "../srcs/fft/CUDA/2D/IFFT/IFFT2D.h"
#endif




// dnn module
#ifdef _decx_dnn
#include "../srcs/nn/operators/operators.h"
#endif




/**
* Ҫͳ�ƴ������������ļ��в��ң���ѡʹ���������ʽ ���� �� ^b*[^:b#/]+.*$
* compute_75,sm_75
* compute_70,sm_70
* compute_60,sm_60
* compute_50,sm_50
* compute_30,sm_30
*/