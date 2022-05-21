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

#include "pch.h"

#include "../srcs/classes/matrix.h"
#include "../srcs/cv/utils/cvt_colors.h"


#include "../srcs/GEMM/CPU/sgemm.h"