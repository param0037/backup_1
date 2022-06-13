#ifndef _CREATING_TENSOR_H_
#define _CREATING_TENSOR_H_

#pragma comment(lib, "../../../bin/x64/DECX_CUDA.lib")
#pragma comment(lib, "../../../bin/x64/DECX_cpu.lib")
#pragma comment(lib, "../../../bin/x64/DECX_allocation.lib")
#pragma comment(lib, "../../../bin/x64/DECX.lib")
#include "../../../APIs/DECX.h"
#include <iostream>
#include <iomanip>


using namespace std;


template <typename T>
void print_cube(de::Tensor<T>* src, const uint x1, const uint x2, const uint y1, const uint y2, 
    const uint z1, const uint z2)
{
    for (int i = x1; i < x2; ++i) {
        for (int j = y1; j < y2; ++j) {
            cout << '[';
            for (int z = z1; z < z2; ++z) {
                cout << setw(3) << src->index(i, j, z) << ',';
            }
            cout << ']';
        }
        cout << endl;
    }
}



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