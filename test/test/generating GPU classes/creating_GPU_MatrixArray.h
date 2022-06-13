#ifndef _CREATING_MATRIXARRAY_H_
#define _CREATING_MATRIXARRAY_H_

#pragma comment(lib, "../../../bin/x64/DECX.lib")
#pragma comment(lib, "../../../bin/x64/DECX_CUDA.lib")
#pragma comment(lib, "../../../bin/x64/DECX_cpu.lib")


#include "../../../APIs/DECX.h"
#include <iostream>
#include <iomanip>


using namespace std;


template <typename T>
void print_multi_plane(de::MatrixArray<T>* src, const uint x1, const uint x2, const uint y1, const uint y2,
    const uint z1, const uint z2)
{
    for (int z = z1; z < z2; ++z) {
        for (int i = x1; i < x2; ++i) {
            for (int j = y1; j < y2; ++j) {
                cout << setw(3) << src->index(i, j, z);
            }
            cout << endl;
        }
        cout << '\n';
    }
}



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
}


#endif