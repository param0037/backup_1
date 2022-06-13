// CUDA_GEMM.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>

#pragma comment(lib, "../../../bin/x64/DECX_CUDA.lib")
#pragma comment(lib, "../../../bin/x64/DECX_cpu.lib")

#include "../../../APIs/DECX.h"
#include <iomanip>
#include <ctime>


using namespace std;

template <int M, int N, int K>
void cuda_gemm_mem_exch()
{
    de::InitCuda();
    de::Matrix<float>& A = de::CreateMatrixRef<float>(K, M, 0);
    de::Matrix<float>& B = de::CreateMatrixRef<float>(N, K, 0);
    de::Matrix<float>& C = de::CreateMatrixRef<float>(N, M, 0);

    for (int i = 0; i < A.Height(); ++i) {
        for (int j = 0; j < A.Width(); ++j) {
            A.index(i, j) = j * 0.1;
        }
    }

    for (int i = 0; i < B.Height(); ++i) {
        for (int j = 0; j < B.Width(); ++j) {
            B.index(i, j) = i * 0.1;
        }
    }

    float sum = 0;

    for (int i = 0; i < K; ++i) {
        sum += (0.1 * i * 0.1 * i);
    }

    cout << "The every results should be :" << sum << endl;
    clock_t s, e;
    s = clock();
    de::cuda::GEMM(A, B, C);
    e = clock();

    cout << "The results are : \n";
    for (int i = C.Height() - 10; i < C.Height(); ++i) {
        for (int j = C.Width() - 10; j < C.Width(); ++j) {
            cout << setw(10) << C.index(i, j);
        }
        cout << endl;
    }
    cout << "time cost : " << (e - s) << endl;
}


template <int M, int N, int K>
void cuda_gemm_on_GPU()
{
    de::InitCuda();
    de::Matrix<float>& A = de::CreateMatrixRef<float>(K, M, 0);
    de::Matrix<float>& B = de::CreateMatrixRef<float>(N, K, 0);
    de::Matrix<float>& C = de::CreateMatrixRef<float>(N, M, 0);

    de::GPU_Matrix<float>& dev_A = de::CreateGPUMatrixRef<float>(K, M);
    de::GPU_Matrix<float>& dev_B = de::CreateGPUMatrixRef<float>(N, K);
    de::GPU_Matrix<float>& dev_C = de::CreateGPUMatrixRef<float>(N, M);

    for (int i = 0; i < A.Height(); ++i) {
        for (int j = 0; j < A.Width(); ++j) {
            A.index(i, j) = j * 0.1;
        }
    }

    for (int i = 0; i < B.Height(); ++i) {
        for (int j = 0; j < B.Width(); ++j) {
            B.index(i, j) = i * 0.1;
        }
    }

    dev_A.Load_from_host(A);
    dev_B.Load_from_host(B);

    float sum = 0;

    for (int i = 0; i < K; ++i) {
        sum += (0.1 * i * 0.1 * i);
    }

    cout << "The every results should be :" << sum << endl;
    clock_t s, e;
    s = clock();
    de::cuda::GEMM(dev_A, dev_B, dev_C);
    e = clock();

    cout << "The results are : \n";

    dev_C.Load_to_host(C);

    for (int i = C.Height() - 10; i < C.Height(); ++i) {
        for (int j = C.Width() - 10; j < C.Width(); ++j) {
            cout << setw(12) << C.index(i, j);
        }
        cout << endl;
    }
    cout << "time cost : " << (e - s) << endl;
}


int main()
{
    //cuda_gemm_mem_exch<2048, 2048, 2048>();
    cuda_gemm_on_GPU<2048, 2048, 2048>();
    return 0;
}