
#pragma comment(lib, "../../../bin/x64/DECX_CUDA.lib")
#pragma comment(lib, "../../../bin/x64/DECX_cpu.lib")
#pragma comment(lib, "../../../bin/x64/DECX_allocation.lib")
#pragma comment(lib, "../../../bin/x64/DECX.lib")
#include "../../../APIs/DECX.h"
#include <iostream>
#include <iomanip>


#define Image_path L"E:/DECX_world/test/test/Image_FFT/test_image.jpg"

int main()
{
    de::InitCuda();

    de::vis::Img& src = de::vis::CreateImgRef();
    de::DH handle = de::vis::ReadImage(Image_path, src);
    if (handle.error_type != de::DECX_SUCCESS) {
        printf(handle.error_string);
        exit(-1);
    }
    de::vis::Img& src_gray = de::vis::CreateImgRef(src.Width(), src.Height(), de::vis::DE_UC1);
    de::vis::Img& dst = de::vis::CreateImgRef(src.Width(), src.Height(), de::vis::DE_UC1);
    de::vis::Img& FFT_res = de::vis::CreateImgRef(src.Width(), src.Height(), de::vis::DE_UC1);

    de::vis::merge_channel(src, src_gray, de::vis::BGR_to_Gray);

    de::Matrix<float>& A = de::CreateMatrixRef<float>(src.Width(), src.Height(), de::DATA_STORE_TYPE::Page_Locked);
    de::Matrix<de::CPf>& B = de::CreateMatrixRef<de::CPf>(src.Width(), src.Height(), de::DATA_STORE_TYPE::Page_Locked);

    for (int i = 0; i < A.Height(); ++i) {
        for (int j = 0; j < A.Width(); ++j) {
            A.index(i, j) = *src_gray.Ptr(i, j);
        }
    }

    de::fft::FFT2D_R2C_f(A, B);

    for (int i = 0; i < A.Height(); ++i) {
        for (int j = 0; j < A.Width(); ++j) {
            *FFT_res.Ptr(i, j) = (uchar)((abs(B.index(i, j).real) + abs(B.index(i, j).image)) / 100);
        }
    }

    de::fft::IFFT2D_C2R_f(B, A);

    for (int i = 0; i < A.Height(); ++i) {
        for (int j = 0; j < A.Width(); ++j) {
            *dst.Ptr(i, j) = (uchar)A.index(i, j);
        }
    }

    de::vis::ShowImage(src_gray, L"original");
    de::vis::ShowImage(FFT_res, L"FFT_result");
    de::vis::ShowImage(dst, L"IFFT_result");
    de::vis::Wait();

    return 0;
}