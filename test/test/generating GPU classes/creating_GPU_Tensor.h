#ifndef _CREATING_TENSOR_H_
#define _CREATING_TENSOR_H_

#pragma comment(lib, "../../../bin/x64/DECX_CUDA.lib")
#pragma comment(lib, "../../../bin/x64/DECX_cpu.lib")
#include "../../../APIs/DECX.h"
#include <iostream>
#include <iomanip>
#include "../utils/printf.h"

using namespace std;



template <typename T>
void generate_GPU_Tensor(const uint W, const uint H, const uint D, const int store_type)
{
    de::Tensor<T>& A = de::CreateTensorRef<T>(W, H, D, store_type);

    de::GPU_Tensor<T>& dev_A = de::CreateGPUTensorRef<T>(W, H, D);

    for (int i = 0; i < A.Height(); i++) {
        for (int j = 0; j < A.Width(); ++j) {
            for (int z = 0; z < A.Depth(); ++z) {
                A.index(i, j, z) = i + j + z;
            }
        }
    }

    dev_A.Load_from_host(A);

    print_cube<T>(&A, 0, 10, 0, 10, 0, 4);

    dev_A.release();
    A.release();

    de::cuda::DECX_CUDA_exit();
}


#endif