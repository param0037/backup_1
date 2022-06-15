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
void generate_Tensor(const uint W, const uint H, const uint D, const int store_type)
{
    de::Tensor<T>& A = de::CreateTensorRef<T>(W, H, D, store_type);

    for (int i = 0; i < A.Height(); i++) {
        for (int j = 0; j < A.Width(); ++j) {
            for (int z = 0; z < A.Depth(); ++z) {
                A.index(i, j, z) = i + j + z;
            }
        }
    }

    print_cube<T>(&A, 0, 10, 0, 10, 0, 4);

    A.release();
}


#endif