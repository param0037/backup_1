
#pragma comment(lib, "../../../bin/x64/DECX_CUDA.lib")
#pragma comment(lib, "../../../bin/x64/DECX_cpu.lib")
#pragma comment(lib, "../../../bin/x64/DECX_allocation.lib")
#pragma comment(lib, "../../../bin/x64/DECX.lib")
#include "../../../APIs/DECX.h"
#include <iostream>
#include <iomanip>

using namespace std;


#define _signal_length_ 50


void FFT1D_R2C()
{
    de::InitCuda();

    de::Vector<float>& A = de::CreateVectorRef<float>(_signal_length_, de::DATA_STORE_TYPE::Page_Locked);
    de::Vector<de::CPf>& B = de::CreateVectorRef<de::CPf>();
    de::CPf tmp;
    de::DH handle;

    for (int i = 0; i < A.Len(); ++i) {
        A.index(i) = i;
    }

    cout << "original vector:\n";
    for (int i = 0; i < A.Len(); ++i) {
        cout << A.index(i) << endl;
    }

    handle = de::fft::FFT1D_R2C_f(A, B);
    if (handle.error_type != de::DECX_SUCCESS) {
        printf(handle.error_string);
        return;
    }

    handle = de::fft::IFFT1D_C2R_f(B, A);
    if (handle.error_type != de::DECX_SUCCESS) {
        printf(handle.error_string);
        return;
    }

    cout << "\nFFT result:\n";
    
    for (int i = 0; i < B.Len(); ++i) {
        tmp = B.index(i);
        cout << tmp.real << ", " << tmp.image << endl;
    }

    cout << "\nIFFT result:\n";
    for (int i = 0; i < A.Len(); ++i) {
        cout << A.index(i) << endl;
    }

    A.release();
    B.release();
}


void FFT1D_C2C()
{
    de::InitCuda();

    de::Vector<de::CPf>& A = de::CreateVectorRef<de::CPf>(_signal_length_, de::DATA_STORE_TYPE::Page_Locked);
    de::Vector<de::CPf>& B = de::CreateVectorRef<de::CPf>();
    de::CPf tmp;
    de::DH handle;

    for (int i = 0; i < A.Len(); ++i) {
        A.index(i).real = i;
    }

    cout << "original vector:\n";
    for (int i = 0; i < A.Len(); ++i) {
        tmp = A.index(i);
        cout << tmp.real << ", " << tmp.image << endl;
    }

    handle = de::fft::FFT1D_C2C_f(A, B);
    if (handle.error_type != de::DECX_SUCCESS) {
        printf(handle.error_string);
        return;
    }

    handle = de::fft::IFFT1D_C2C_f(B, A);
    if (handle.error_type != de::DECX_SUCCESS) {
        printf(handle.error_string);
        return;
    }

    cout << "\nFFT result:\n";
    
    for (int i = 0; i < B.Len(); ++i) {
        tmp = B.index(i);
        cout << tmp.real << ", " << tmp.image << endl;
    }

    cout << "\nIFFT result:\n";
    for (int i = 0; i < A.Len(); ++i) {
        tmp = A.index(i);
        cout << tmp.real << ", " << tmp.image << endl;
    }

    A.release();
    B.release();
}


int main()
{
    FFT1D_R2C();
    //FFT1D_C2C();

    return 0;
}