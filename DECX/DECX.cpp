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


#define _decx_cv
// cv module
#ifdef _decx_cv


//#include "srcs/cv/Gaussian filter/Gaussian_Filter.h"
//#include "srcs/cv/NLM/NLM.h"

//#include "srcs/cv/Canny/Canny.h"

#include "../srcs/cv/cv_classes/cv_classes.h"
#include "../srcs/cv/cv_classes/cv_cls_MFuncs.h"

//#include "srcs/cv/Convert2MatAndTens/Convert.h"

#include "../srcs/cv/Image_IO_GUI/IO.h"
#include "../srcs/cv/Image_IO_GUI/GUI.h"
#endif