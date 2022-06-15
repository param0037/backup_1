#ifndef _CREATING_TENSORARRAY_H_
#define _CREATING_TENSORARRAY_H_

#pragma comment(lib, "../../../bin/x64/DECX_CUDA.lib")
#pragma comment(lib, "../../../bin/x64/DECX_cpu.lib")
#include "../../../APIs/DECX.h"
#include <iostream>
#include <iomanip>
#include "../utils/printf.h"

using namespace std;



template <typename T>
void generate_TensorArray(const uint W, const uint H, const uint D, const uint TensorNum, const int store_type)
{
    de::TensorArray<T>& A = de::CreateTensorArrayRef<T>(W, H, D, TensorNum, store_type);

    for (int w = 0; w < A.TensorNum(); ++w) {
        for (int i = 0; i < A.Height(); i++) {
            for (int j = 0; j < A.Width(); ++j) {
                for (int z = 0; z < A.Depth(); ++z) {
                    A.index(i, j, z, w) = i + j + z + w;
                }
            }
        }
    }
    print_tesseract<T>(&A, 0, 10, 0, 10, 0, 4, 0, 2);

    A.release();
}


#endif