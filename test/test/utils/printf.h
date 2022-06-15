#ifndef _PRINTF_H_
#define _PRINTF_H_

#pragma comment(lib, "../../../bin/x64/DECX_CUDA.lib")
#pragma comment(lib, "../../../bin/x64/DECX_cpu.lib")
#include "../../../APIs/DECX.h"
#include <iostream>
#include <iomanip>


using namespace std;

template <typename T>
void print_square(de::Matrix<T>* src, const uint x1, const uint x2, const uint y1, const uint y2)
{
    for (int i = x1; i < x2; ++i) {
        for (int j = y1; j < y2; ++j) {
            cout << setw(5) << src->index(i, j);
        }
        cout << endl;
    }
}


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
void print_line(de::Vector<T>* src, const uint x1, const uint x2)
{
    for (int i = x1; i < x2; ++i) {
        cout << setw(5) << src->index(i);
    }
}


#endif
