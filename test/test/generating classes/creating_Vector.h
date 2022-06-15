#ifndef _CREATING_VECTOR_H_
#define _CREATING_VECTOR_H_

#pragma comment(lib, "../../../bin/x64/DECX_CUDA.lib")
#pragma comment(lib, "../../../bin/x64/DECX_cpu.lib")
#include "../../../APIs/DECX.h"
#include <iostream>
#include <iomanip>
#include "../utils/printf.h"

using namespace std;



template <typename T>
void generate_Vector(const size_t length, const int store_type)
{
    de::Vector<T>& A = de::CreateVectorRef<T>(length, store_type);

    for (size_t i = 0; i < A.Len(); i++) {
         A.index(i) = i;
    }

    print_line<T>(&A, 0, 10);

    A.release();
}


#endif