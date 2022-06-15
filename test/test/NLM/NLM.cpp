// NLM.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#pragma comment(lib, "../../../bin/x64/DECX_CUDA.lib")
#pragma comment(lib, "../../../bin/x64/DECX_cpu.lib")
#pragma comment(lib, "../../../bin/x64/Image_IO_GUI.lib")
#include "../../../APIs/DECX.h"
#include <iostream>
#include <iomanip>

using namespace std;

void NLM_colored_keep_alpha()
{
	de::vis::Img& src = de::vis::CreateImgRef();

	std::string tmp;
	cout << "drag image here\n";
	cin >> tmp;
	de::vis::ReadImage(tmp, src);
	
	de::vis::Img& dst = de::vis::CreateImgRef(src.Width(), src.Height(), de::vis::DE_UC4);
	clock_t s, e;
	s = clock();
	de::vis::cuda::NLM_RGB_keep_alpha(src, dst, 7, 1, 15);
	e = clock();

	cout << "Time cost : " << e - s << " msec\n";

	de::vis::ShowImage(src, L"src");
	de::vis::ShowImage(dst, L"dst");
	de::vis::Wait();
}



void NLM_colored()
{
	de::vis::Img& src = de::vis::CreateImgRef();

	std::string tmp;
	cout << "drag image here\n";
	cin >> tmp;
	de::vis::ReadImage(tmp, src);

	de::vis::Img& dst = de::vis::CreateImgRef(src.Width(), src.Height(), de::vis::DE_UC4);
	clock_t s, e;
	s = clock();
	de::vis::cuda::NLM_RGB(src, dst, 7, 1, 15);
	e = clock();

	cout << "Time cost : " << e - s << " msec\n";

	de::vis::ShowImage(src, L"src");
	de::vis::ShowImage(dst, L"dst");
	de::vis::Wait();
}



void NLM_gray()
{
	de::vis::Img& src = de::vis::CreateImgRef();

	std::string tmp;
	cout << "drag image here\n";
	cin >> tmp;
	de::vis::ReadImage(tmp, src);

	de::vis::Img& src_gray = de::vis::CreateImgRef(src.Width(), src.Height(), de::vis::DE_UC1);
	de::vis::merge_channel(src, src_gray, de::vis::BGR_to_Gray);
	
	de::vis::Img& dst = de::vis::CreateImgRef(src.Width(), src.Height(), de::vis::DE_UC1);
	clock_t s, e;
	s = clock();
	de::vis::cuda::NLM_Gray(src_gray, dst, 7, 1, 15);
	e = clock();

	cout << "Time cost : " << e - s << " msec\n";

	de::vis::ShowImage(src_gray, L"src");
	de::vis::ShowImage(dst, L"dst");
	de::vis::Wait();
}



int main()
{
	de::InitCuda();
	NLM_colored_keep_alpha();
	//NLM_gray();
	//NLM_colored();

	return 0;
}
