#ifndef _CREATING_MATRIX_H_
#define _CREATING_MATRIX_H_

#pragma comment(lib, "../../../bin/x64/DECX_CUDA.lib")
#pragma comment(lib, "../../../bin/x64/DECX_cpu.lib")
#pragma comment(lib, "../../../bin/x64/DECX_allocation.lib")
#pragma comment(lib, "../../../bin/x64/DECX.lib")
#include "../../../APIs/DECX.h"
#include <iostream>
#include <iomanip>


using namespace std;


template <typename T>
void print_square(de::Matrix<T> *src, const uint x1, const uint x2, const uint y1, const uint y2)
{
    for (int i = x1; i < x2; ++i) {
        for (int j = y1; j < y2; ++j) {
            cout << setw(5) << src->index(i, j);
        }
        cout << endl;
    }
}



template <typename T>
void generate_GPU_matrix(const uint W, const uint H, const int store_type)
{
    de::Matrix<T>& A = de::CreateMatrixRef<T>(W, H, store_type);

    de::GPU_Matrix<T>& dev_A = de::CreateGPUMatrixRef<T>(W, H);

    for (int i = 0; i < A.Height(); i++) {
        for (int j = 0; j < A.Width(); ++j) {
            A.index(i, j) = i + j;
        }
    }

    dev_A.Load_from_host(A);

    for (int i = 0; i < A.Height(); i++) {
        for (int j = 0; j < A.Width(); ++j) {
            A.index(i, j) = 0;
        }
    }

    dev_A.Load_to_host(A);

    print_square<T>(&A, 0, 10, 0, 10);

    dev_A.release();
    A.release();
}


#endif