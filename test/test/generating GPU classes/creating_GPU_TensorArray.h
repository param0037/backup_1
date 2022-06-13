#ifndef _CREATING_TENSORARRAY_H_
#define _CREATING_TENSORARRAY_H_

#pragma comment(lib, "../../../bin/x64/DECX_CUDA.lib")
#pragma comment(lib, "../../../bin/x64/DECX_cpu.lib")
#pragma comment(lib, "../../../bin/x64/DECX_allocation.lib")
#pragma comment(lib, "../../../bin/x64/DECX.lib")
#include "../../../APIs/DECX.h"
#include <iostream>
#include <iomanip>


using namespace std;


template <typename T>
void print_tesseract(de::TensorArray<T>* src, const uint x1, const uint x2, const uint y1, const uint y2,
    const uint z1, const uint z2, const uint w1, const uint w2)
{
    for (int w = w1; w < w2; ++w) {
        for (int i = x1; i < x2; ++i) {
            for (int j = y1; j < y2; ++j) {
                cout << '[';
                for (int z = z1; z < z2; ++z) {
                    cout << setw(3) << src->index(i, j, z, w) << ',';
                }
                cout << ']';
            }
            cout << endl;
        }
        cout << endl;
    }
}



template <typename T>
void generate_TensorArray(const uint W, const uint H, const uint D, const uint TensorNum, const int store_type)
{
    de::TensorArray<T>& A = de::CreateTensorArrayRef<T>(W, H, D, TensorNum, store_type);

    de::GPU_TensorArray<T>& dev_A = de::CreateGPUTensorArrayRef<T>(W, H, D, TensorNum);

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

    cout << "next:\n";

    dev_A.Load_from_host(A);

    for (int w = 0; w < A.TensorNum(); ++w) {
        for (int i = 0; i < A.Height(); i++) {
            for (int j = 0; j < A.Width(); ++j) {
                for (int z = 0; z < A.Depth(); ++z) {
                    A.index(i, j, z, w) = 0;
                }
            }
        }
    }

    dev_A.Load_to_host(A);

    print_tesseract<T>(&A, 0, 10, 0, 10, 0, 4, 0, 2);

    A.release();
}


#endif