#ifndef _CREATING_MATRIX_H_
#define _CREATING_MATRIX_H_

#pragma comment(lib, "../../../bin/x64/DECX_CUDA.lib")
#pragma comment(lib, "../../../bin/x64/DECX_cpu.lib")
#include "../../../APIs/DECX.h"
#include <iostream>
#include <iomanip>
#include "../utils/printf.h"


using namespace std;





template <typename T>
void generate_matrix(const uint W, const uint H, const int store_type)
{
    // Create a de::Matrix<typename T> on host, with width of W and height of H
    de::Matrix<T>& A = de::CreateMatrixRef<T>(W, H, store_type);

    // assign values for matrix A
    for (int i = 0; i < A.Height(); i++) {
        for (int j = 0; j < A.Width(); ++j) {
            A.index(i, j) = i + j;
        }
    }

    // print the values in it
    print_square<T>(&A, 0, 10, 0, 10);

    // release data space of matrix A on RAM
    A.release();
}


#endif