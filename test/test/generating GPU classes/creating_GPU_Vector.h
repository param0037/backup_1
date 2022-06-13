#ifndef _CREATING_VECTOR_H_
#define _CREATING_VECTOR_H_

#pragma comment(lib, "../../../bin/x64/DECX_CUDA.lib")
#pragma comment(lib, "../../../bin/x64/DECX_cpu.lib")
#pragma comment(lib, "../../../bin/x64/DECX_allocation.lib")
#pragma comment(lib, "../../../bin/x64/DECX.lib")
#include "../../../APIs/DECX.h"
#include <iostream>
#include <iomanip>


using namespace std;


template <typename T>
void print_line(de::Vector<T>* src, const uint x1, const uint x2)
{
    for (int i = x1; i < x2; ++i) {
        cout << setw(5) << src->index(i);
    }
}



template <typename T>
void generate_GPU_Vector(const size_t length, const int store_type)
{
    de::Vector<T>& A = de::CreateVectorRef<T>(length, store_type);

    de::GPU_Vector<T>& dev_A = de::CreateGPUVectorRef<T>(length);

    for (size_t i = 0; i < A.Len(); i++) {
         A.index(i) = i;
    }

    dev_A.load_from_host(A);

    for (size_t i = 0; i < A.Len(); i++) {
        A.index(i) = 0;
    }

    dev_A.load_to_host(A);

    print_line<T>(&A, 0, 10);

    dev_A.release();
    A.release();
}


#endif