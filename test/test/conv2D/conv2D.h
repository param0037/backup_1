#include <iostream>

#pragma comment(lib, "../../../bin/x64/DECX_CUDA.lib")
#pragma comment(lib, "../../../bin/x64/DECX_cpu.lib")
#pragma comment(lib, "../../../bin/x64/Image_IO_GUI.lib")

#include "../../../APIs/DECX.h"
#include <iomanip>
#include <ctime>

using namespace std;

void conv2()
{
	de::InitCuda();
	de::vis::Img& src = de::vis::CreateImgRef();

	std::string tmp;
	cout << "drag image here\n";
	cin >> tmp;
	de::vis::ReadImage(tmp, src);

	de::vis::Img& gray = de::vis::CreateImgRef(src.Width(), src.Height(), de::vis::ImgConstructType::DE_UC1);
	de::vis::merge_channel(src, gray, de::vis::ImgChannelMergeType::BGR_to_Gray);

	de::vis::ShowImage(gray, L"1");

	de::Matrix<float>& A = de::CreateMatrixRef<float>(src.Width(), src.Height(), 0);

	de::Matrix<float>& B = de::CreateMatrixRef<float>(src.Width() - 4, src.Height() - 4, 0);

	de::Matrix<float>& kernel = de::CreateMatrixRef<float>(5, 5, 0);

	// set the kernel
	for (int i = 0; i < kernel.Height(); ++i) {
		for (int j = 0; j < kernel.Width(); ++j) {
			kernel.index(i, j) = 1.0f / (kernel.Width() * kernel.Height());
		}
	}

	// load pixels
	for (int i = 0; i < A.Height(); ++i) {
		for (int j = 0; j < A.Width(); ++j) {
			A.index(i, j) = *gray.Ptr(i, j);
		}
	}

	clock_t s, e;
	s = clock();
	//de::DH handle = de::cuda::Conv2(A, kernel, B, de::conv_property::de_conv_zero_compensate);
	de::DH handle = de::cuda::Conv2(A, kernel, B, de::conv_property::de_conv_no_compensate);
	e = clock();
	if (handle.error_type != de::DECX_SUCCESS) {
		cout << handle.error_string << endl;
		return;
	}

	de::vis::Img& dst = de::vis::CreateImgRef(B.Width(), B.Height(), de::vis::ImgConstructType::DE_UC1);

	for (int i = 0; i < B.Height(); ++i) {
		for (int j = 0; j < B.Width(); ++j) {
			*(dst.Ptr(i, j) + 0) = B.index(i, j);
		}
	}
	cout << "time cost : " << e - s << " msec\n";

	de::vis::ShowImage(dst, L"2");
	de::vis::Wait();

	de::cuda::DECX_CUDA_exit();
}



void dev_conv2()
{
	de::InitCuda();
	de::vis::Img& src = de::vis::CreateImgRef();

	std::string tmp;
	cout << "drag image here\n";
	cin >> tmp;
	de::vis::ReadImage(tmp, src);

	de::vis::Img& gray = de::vis::CreateImgRef(src.Width(), src.Height(), de::vis::ImgConstructType::DE_UC1);
	de::vis::merge_channel(src, gray, de::vis::ImgChannelMergeType::BGR_to_Gray);

	de::vis::ShowImage(gray, L"1");

	de::Matrix<float>& A = de::CreateMatrixRef<float>(src.Width(), src.Height(), 0);
	de::GPU_Matrix<float>& dev_A = de::CreateGPUMatrixRef<float>(src.Width(), src.Height());

	de::GPU_Matrix<float>& dev_B = de::CreateGPUMatrixRef<float>();

	de::Matrix<float>& kernel = de::CreateMatrixRef<float>(5, 5, 0);
	de::GPU_Matrix<float>& dev_kernel = de::CreateGPUMatrixRef<float>(5, 5);

	// set the kernel
	for (int i = 0; i < kernel.Height(); ++i) {
		for (int j = 0; j < kernel.Width(); ++j) {
			kernel.index(i, j) = 1.0f / (kernel.Width() * kernel.Height());
		}
	}

	// load pixels
	for (int i = 0; i < A.Height(); ++i) {
		for (int j = 0; j < A.Width(); ++j) {
			A.index(i, j) = *gray.Ptr(i, j);
		}
	}

	dev_A.Load_from_host(A);
	dev_kernel.Load_from_host(kernel);

	clock_t s, e;
	s = clock();
	//de::DH handle = de::cuda::Conv2(dev_A, dev_kernel, dev_B, de::conv_property::de_conv_zero_compensate);
	de::DH handle = de::cuda::Conv2(dev_A, dev_kernel, dev_B, de::conv_property::de_conv_no_compensate);
	e = clock();
	if (handle.error_type != de::DECX_SUCCESS) {
		cout << handle.error_string << endl;
		return;
	}

	de::Matrix<float>& B = de::CreateMatrixRef<float>(dev_B.Width(), dev_B.Height(), 0);
	dev_B.Load_to_host(B);

	cout << handle.error_string << endl;

	de::vis::Img& dst = de::vis::CreateImgRef(B.Width(), B.Height(), de::vis::ImgConstructType::DE_UC1);

	for (int i = 0; i < B.Height(); ++i) {
		for (int j = 0; j < B.Width(); ++j) {
			*(dst.Ptr(i, j) + 0) = B.index(i, j);
		}
	}

	cout << "time cost : " << e - s << " msec\n";

	de::vis::ShowImage(dst, L"2");
	de::vis::Wait();

	de::cuda::DECX_CUDA_exit();
}



void dev_conv2_fp16()
{
	de::InitCuda();
	de::vis::Img& src = de::vis::CreateImgRef();

	std::string tmp;
	cout << "drag image here\n";
	cin >> tmp;
	de::vis::ReadImage(tmp, src);

	de::vis::Img& gray = de::vis::CreateImgRef(src.Width(), src.Height(), de::vis::ImgConstructType::DE_UC1);
	de::vis::merge_channel(src, gray, de::vis::ImgChannelMergeType::BGR_to_Gray);

	de::vis::ShowImage(gray, L"1");

	de::Matrix<de::Half>& A = de::CreateMatrixRef<de::Half>(src.Width(), src.Height(), 0);
	de::GPU_Matrix<de::Half>& dev_A = de::CreateGPUMatrixRef<de::Half>(src.Width(), src.Height());

	de::GPU_Matrix<de::Half>& dev_B = de::CreateGPUMatrixRef<de::Half>();

	de::Matrix<de::Half>& kernel = de::CreateMatrixRef<de::Half>(5, 5, 0);
	de::GPU_Matrix<de::Half>& dev_kernel = de::CreateGPUMatrixRef<de::Half>(5, 5);

	// set the kernel
	for (int i = 0; i < kernel.Height(); ++i) {
		for (int j = 0; j < kernel.Width(); ++j) {
			kernel.index(i, j) = de::Float2Half(1.0f / (kernel.Width() * kernel.Height()));
		}
	}

	// load pixels
	for (int i = 0; i < A.Height(); ++i) {
		for (int j = 0; j < A.Width(); ++j) {
			A.index(i, j) = de::Float2Half(*gray.Ptr(i, j));
		}
	}

	dev_A.Load_from_host(A);
	dev_kernel.Load_from_host(kernel);

	clock_t s, e;
	s = clock();
	//de::DH handle = de::cuda::Conv2(dev_A, dev_kernel, dev_B, de::conv_property::de_conv_zero_compensate);
	de::DH handle = de::cuda::Conv2(dev_A, dev_kernel, dev_B, de::conv_property::de_conv_no_compensate);
	e = clock();
	if (handle.error_type != de::DECX_SUCCESS) {
		cout << handle.error_string << endl;
		return;
	}

	de::Matrix<de::Half>& B = de::CreateMatrixRef<de::Half>(dev_B.Width(), dev_B.Height(), 0);
	dev_B.Load_to_host(B);

	cout << handle.error_string << endl;

	de::vis::Img& dst = de::vis::CreateImgRef(B.Width(), B.Height(), de::vis::ImgConstructType::DE_UC1);

	for (int i = 0; i < B.Height(); ++i) {
		for (int j = 0; j < B.Width(); ++j) {
			*(dst.Ptr(i, j) + 0) = de::Half2Float(B.index(i, j));
		}
	}

	cout << "time cost : " << e - s << " msec\n";

	de::vis::ShowImage(dst, L"2");
	de::vis::Wait();

	de::cuda::DECX_CUDA_exit();
}



void dev_conv2_mk()
{
	de::InitCuda();

	de::vis::Img& src = de::vis::CreateImgRef();
	std::string tmp;
	cout << "drag image here\n";
	cin >> tmp;
	de::vis::ReadImage(tmp, src);
	
	de::MatrixArray<float>& A = de::CreateMatrixArrayRef<float>(src.Width(), src.Height(), 4, 0);
	de::GPU_MatrixArray<float>& dev_A = de::CreateGPUMatrixArrayRef<float>(src.Width(), src.Height(), 4);

	de::GPU_MatrixArray<float>& dev_B = de::CreateGPUMatrixArrayRef<float>();

	de::MatrixArray<float>& kernel = de::CreateMatrixArrayRef<float>(3 * 2 + 1, 4 * 2 + 1, 4, 0);
	de::GPU_MatrixArray<float>& dev_kernel = de::CreateGPUMatrixArrayRef<float>(3 * 2 + 1, 4 * 2 + 1, 4);

	for (int i = 0; i < kernel.Height(); ++i) {
		for (int j = 0; j < kernel.Width(); ++j) {
			kernel.index(i, j, 0) = 1.0f / (kernel.Width() * kernel.Height());	// B
			kernel.index(i, j, 1) = 1.0f / (kernel.Width() * kernel.Height());	// G
			kernel.index(i, j, 2) = 1.0f / (kernel.Width() * kernel.Height());	// R
		}
	}

	for (int i = 0; i < A.Height(); ++i) {
		for (int j = 0; j < A.Width(); ++j) {
			A.index(i, j, 0) = *src.Ptr(i, j);
			A.index(i, j, 1) = *(src.Ptr(i, j) + 1);
			A.index(i, j, 2) = *(src.Ptr(i, j) + 2);
			A.index(i, j, 3) = *(src.Ptr(i, j) + 3);
		}
	}

	dev_A.Load_from_host(A);
	dev_kernel.Load_from_host(kernel);

	clock_t s, e;
	s = clock();
	//de::DH handle = de::cuda::Conv2_multi_kernel(dev_A, dev_kernel, dev_B, de::conv_property::de_conv_no_compensate);
	de::DH handle = de::cuda::Conv2_multi_kernel(dev_A, dev_kernel, dev_B, de::conv_property::de_conv_zero_compensate);
	e = clock();

	de::MatrixArray<float>& B = de::CreateMatrixArrayRef<float>(dev_B.Width(), dev_B.Height(), dev_B.MatrixNumber(), 0);
	dev_B.Load_to_host(B);

	de::vis::Img& dst = de::vis::CreateImgRef(B.Width(), B.Height(), de::vis::ImgConstructType::DE_UC4);

	for (int i = 0; i < B.Height(); ++i) {
		for (int j = 0; j < B.Width(); ++j) {
			*(dst.Ptr(i, j) + 0) = B.index(i, j, 0);
			*(dst.Ptr(i, j) + 1) = B.index(i, j, 1);
			*(dst.Ptr(i, j) + 2) = B.index(i, j, 2);
			*(dst.Ptr(i, j) + 3) = 255;
		}
	}

	cout << "time cost : " << e - s << " msec\n";

	de::vis::ShowImage(dst, L"2");
	de::vis::Wait();

	de::cuda::DECX_CUDA_exit();
}