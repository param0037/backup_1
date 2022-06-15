#ifndef _CREATING_MATRIXARRAY_H_
#define _CREATING_MATRIXARRAY_H_

#pragma comment(lib, "../../../bin/x64/DECX.lib")
#pragma comment(lib, "../../../bin/x64/DECX_CUDA.lib")


#include "../../../APIs/DECX.h"
#include <iostream>
#include <iomanip>
#include "../utils/printf.h"

using namespace std;



template <typename T>
void generate_GPU_MatrixArray(const uint W, const uint H, const uint Num, const int store_type)
{
    de::MatrixArray<T>& A = de::CreateMatrixArrayRef<T>(W, H, Num, store_type);

    de::GPU_MatrixArray<T>& dev_A = de::CreateGPUMatrixArrayRef<T>(W, H, Num);

    for (int i = 0; i < A.Height(); i++) {
        for (int j = 0; j < A.Width(); ++j) {
            for (int z = 0; z < A.MatrixNumber(); ++z) {
                A.index(i, j, z) = i + j + z;
            }
        }
    }

    dev_A.Load_from_host(A);

    for (int i = 0; i < A.Height(); i++) {
        for (int j = 0; j < A.Width(); ++j) {
            for (int z = 0; z < A.MatrixNumber(); ++z) {
                A.index(i, j, z) = 0;
            }
        }
    }

    dev_A.Load_to_host(A);

    print_multi_plane<T>(&A, 0, 10, 0, 10, 0, 4);

    dev_A.release();
    A.release();

    de::cuda::DECX_CUDA_exit();
}


#endif