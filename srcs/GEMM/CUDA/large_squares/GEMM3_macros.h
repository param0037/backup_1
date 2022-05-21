#pragma once

#include "../../../classes/core_types.h"
#include "../../../core/defines.h"
#include "GEMM.cuh"



namespace decx
{
    /**
    * @param DA, DB, Ddst are the device memories with dimensions that already fit in the kernel demands
    * --x128 aligned on both hA and wB and x16 aligned on K (wA = hB)
    * @param pitch_A(x16 aligned), pitch_B(x128 aligned) are the widths of DA and DB (true widths) (in float)
    * @param hA is the height of DA (x128 aligned)
    */
    static void sGEMM_part(float* DA, float* DB, float* Ddst,
            const int pitch_A, const int pitch_B, const int hA, cudaStream_t* S);


    /**
    * @param DA, DB, Ddst are the device memories with dimensions that already fit in the kernel demands
    * --x128 aligned on both hA and wB and x16 aligned on K (wA = hB)
    * @param pitch_A(x16 aligned), pitch_B(x128 aligned) are the widths of DA and DB (true widths) (in float)
    * @param hA is the height of DA (x128 aligned)
    */
    static void sGEMM_part_ABC(float* DA, float* DB, float* DC, float* Ddst,
            const int pitch_A, const int pitch_B, const int hA, cudaStream_t* S);



    /**
    * @param DA, DB, Ddst are the device memories with dimensions that already fit in the kernel demands
    * --x128 aligned on both hA and wB and x16 aligned on K (wA = hB)
    * @param pitch_A, pitch_B are the widths of DA and DB (true widths) (in float)
    * @param hA is the height of DA (x128 aligned)
    */
    static void hGEMM_part(de::Half* DA, de::Half* DB, de::Half* Ddst,
            const int pitch_A, const int pitch_B, const int hA, cudaStream_t* S);


    /**
    * @param DA, DB, Ddst are the device memories with dimensions that already fit in the kernel demands
    * --x128 aligned on both hA and wB and x16 aligned on K (wA = hB)
    * @param pitch_A, pitch_B are the widths of DA and DB (true widths) (in float)
    * @param hA is the height of DA (x128 aligned)
    */
    static void hGEMM_part_ABC(de::Half* DA, de::Half* DB, de::Half* DC, de::Half* Ddst,
            const int pitch_A, const int pitch_B, const int hA, cudaStream_t* S);
}



static
void decx::sGEMM_part(float* DA, float* DB, float* Ddst,
    const int pitch_A, const int pitch_B, const int hA, cudaStream_t* S)
{
    int threads = GEMM_BlockDim * GEMM_BlockDim;
    dim3 grid(hA / (GEMM_BlockDim * 8), pitch_B / (GEMM_BlockDim * 8));

    const uint __iter = pitch_A / GEMM_BlockDim;

    cu_GEMM_128NT_in << <grid, threads, 0, *S >> > (
        reinterpret_cast<float4*>(DA),
        reinterpret_cast<float4*>(DB),
        reinterpret_cast<float4*>(Ddst), pitch_A / 4, pitch_B / 4, pitch_B / 4, __iter);
}




static
void decx::sGEMM_part_ABC(float* DA, float* DB, float* DC, float* Ddst,
    const int pitch_A, const int pitch_B, const int hA, cudaStream_t* S)
{
    int threads = GEMM_BlockDim * GEMM_BlockDim;
    dim3 grid(hA / (GEMM_BlockDim * 8), pitch_B / (GEMM_BlockDim * 8));

    const uint __iter = pitch_A / GEMM_BlockDim;

    cu_GEMM_128NT_ABC << <grid, threads, 0, *S >> > (
        reinterpret_cast<float4*>(DA),
        reinterpret_cast<float4*>(DB),
        reinterpret_cast<float4*>(DC),
        reinterpret_cast<float4*>(Ddst), pitch_A / 4, pitch_B / 4, pitch_B / 4, __iter);
}




static
void decx::hGEMM_part(de::Half* DA, de::Half* DB, de::Half* Ddst,
    const int pitch_A, const int pitch_B, const int hA, cudaStream_t* S)
{
    int threads = GEMM_BlockDim * GEMM_BlockDim;
    dim3 grid(hA / (GEMM_BlockDim * 8), pitch_B / (GEMM_BlockDim * 8));

    const int __iter = pitch_A / GEMM_BlockDim;

    cu_GEMM_128NT_fp16_in << <grid, threads, 0, *S >> > (
        reinterpret_cast<float4*>(DA),
        reinterpret_cast<float4*>(DB),
        reinterpret_cast<float4*>(Ddst), pitch_A / 8, pitch_B / 8, pitch_B / 8, __iter);
}




static
void decx::hGEMM_part_ABC(de::Half* DA, de::Half* DB, de::Half *DC, de::Half* Ddst,
    const int pitch_A, const int pitch_B, const int hA, cudaStream_t* S)
{
    int threads = GEMM_BlockDim * GEMM_BlockDim;
    dim3 grid(hA / (GEMM_BlockDim * 8), pitch_B / (GEMM_BlockDim * 8));

    const int __iter = pitch_A / GEMM_BlockDim;

    cu_GEMM_128NT_fp16_ABC << <grid, threads, 0, *S >> > (
        reinterpret_cast<float4*>(DA),
        reinterpret_cast<float4*>(DB),
        reinterpret_cast<float4*>(DC),
        reinterpret_cast<float4*>(Ddst), pitch_A / 8, pitch_B / 8, pitch_B / 8, __iter);
}





#define GEMM3_part(__type, _kernel_name) {                                                                                    \
checkCudaErrors(cudaMemcpy2DAsync(d_mem[0].mem,                                                                                \
pitch_A * sizeof(__type), _A->MatptrArr.ptr[0], _A->pitch * sizeof(__type),                                                    \
_A->width * sizeof(__type), _A->height, cudaMemcpyHostToDevice, S[1]));                                                        \
                                                                                                                            \
checkCudaErrors(cudaMemcpy2DAsync(d_mem[1].mem,                                                                                \
    pitch_B * sizeof(__type), _B->MatptrArr.ptr[0], _B->pitch * sizeof(__type),                                                \
    _B->width * sizeof(__type), _B->height, cudaMemcpyHostToDevice, S[1]));                                                    \
                                                                                                                            \
d_mem[0].leading = true;                                                                                                    \
d_mem[1].leading = true;                                                                                                    \
                                                                                                                            \
d_mem[3].leading = false;                                                                                                    \
d_mem[4].leading = false;                                                                                                    \
checkCudaErrors(cudaDeviceSynchronize());                                                                                    \
                                                                                                                            \
for (int i = 0; i < _A->ArrayNumber; ++i)                                                                                    \
{                                                                                                                            \
    if (i > 0) {                                                                                                            \
        if (d_mem[2].leading) {                                                                                                \
            checkCudaErrors(cudaMemcpy2DAsync(_dst->MatptrArr.ptr[i - 1],                                                    \
                _dst->pitch * sizeof(__type), d_mem[2].mem, pitch_B * sizeof(__type), _dst->width * sizeof(__type),            \
                _dst->height, cudaMemcpyDeviceToHost, S[2]));                                                                \
            d_mem[2]._using = true;                                                                                            \
        }                                                                                                                    \
        else {                                                                                                                \
            checkCudaErrors(cudaMemcpy2DAsync(_dst->MatptrArr.ptr[i - 1],                                                    \
                _dst->pitch * sizeof(__type), d_mem[5].mem, pitch_B * sizeof(__type), _dst->width * sizeof(__type),            \
                _dst->height, cudaMemcpyDeviceToHost, S[2]));                                                                \
            d_mem[5]._using = true;                                                                                            \
        }                                                                                                                    \
    }                                                                                                                        \
                                                                                                                            \
    if (d_mem[0].leading) {                                                                                                    \
        _kernel_name(d_mem[0].mem, d_mem[1].mem, d_mem[2].mem, pitch_A, pitch_B, hA, &S[0]);                                \
        d_mem[0]._using = true;                                                                                                \
        d_mem[1]._using = true;                                                                                                \
        d_mem[2]._using = true;                                                                                                \
                                                                                                                            \
        d_mem[2].leading = true;                                                                                            \
        d_mem[5].leading = false;                                                                                            \
    }                                                                                                                        \
    else {                                                                                                                    \
        _kernel_name(d_mem[3].mem, d_mem[4].mem, d_mem[5].mem, pitch_A, pitch_B, hA, &S[0]);                                \
        d_mem[3]._using = true;                                                                                                \
        d_mem[4]._using = true;                                                                                                \
        d_mem[5]._using = true;                                                                                                \
                                                                                                                            \
        d_mem[5].leading = true;                                                                                            \
        d_mem[2].leading = false;                                                                                            \
    }                                                                                                                        \
                                                                                                                            \
    if (i < _A->ArrayNumber - 1) {                                                                                            \
        if (!d_mem[0]._using) {                                                                                                \
            checkCudaErrors(cudaMemcpy2DAsync(d_mem[0].mem,                                                                    \
                pitch_A * sizeof(__type), _A->MatptrArr.ptr[i + 1], _A->pitch * sizeof(__type),                                \
                _A->width * sizeof(__type), _A->height, cudaMemcpyHostToDevice, S[1]));                                        \
                                                                                                                            \
            checkCudaErrors(cudaMemcpy2DAsync(d_mem[1].mem,                                                                    \
                pitch_B * sizeof(__type), _B->MatptrArr.ptr[i + 1], _B->pitch * sizeof(__type),                                \
                _B->width * sizeof(__type), _B->height, cudaMemcpyHostToDevice, S[1]));                                        \
                                                                                                                            \
            d_mem[0].leading = true;                                                                                        \
            d_mem[1].leading = true;                                                                                        \
                                                                                                                            \
            d_mem[3].leading = false;                                                                                        \
            d_mem[4].leading = false;                                                                                        \
        }                                                                                                                    \
        else {                                                                                                                \
            checkCudaErrors(cudaMemcpy2DAsync(d_mem[3].mem,                                                                    \
                pitch_A * sizeof(__type), _A->MatptrArr.ptr[i + 1], _A->pitch * sizeof(__type),                                \
                _A->width * sizeof(__type), _A->height, cudaMemcpyHostToDevice, S[1]));                                        \
                                                                                                                            \
            checkCudaErrors(cudaMemcpy2DAsync(d_mem[4].mem,                                                                    \
                pitch_B * sizeof(__type), _B->MatptrArr.ptr[i + 1], _B->pitch * sizeof(__type),                                \
                _B->width * sizeof(__type), _B->height, cudaMemcpyHostToDevice, S[1]));                                        \
                                                                                                                            \
            d_mem[3].leading = true;                                                                                        \
            d_mem[4].leading = true;                                                                                        \
                                                                                                                            \
            d_mem[0].leading = false;                                                                                        \
            d_mem[1].leading = false;                                                                                        \
        }                                                                                                                    \
    }                                                                                                                        \
                                                                                                                            \
    checkCudaErrors(cudaDeviceSynchronize());                                                                                \
                                                                                                                            \
    d_mem[0]._using = false;                                                                                                \
    d_mem[1]._using = false;                                                                                                \
    d_mem[2]._using = false;                                                                                                \
    d_mem[3]._using = false;                                                                                                \
    d_mem[4]._using = false;                                                                                                \
    d_mem[5]._using = false;                                                                                                \
}                                                                                                                            \
                                                                                                                            \
if (d_mem[2].leading) {                                                                                                        \
    checkCudaErrors(cudaMemcpy2DAsync(_dst->MatptrArr.ptr[_A->ArrayNumber - 1],                                                \
        _dst->pitch * sizeof(__type), d_mem[2].mem, pitch_B * sizeof(__type), _dst->width * sizeof(__type),                    \
        _dst->height, cudaMemcpyDeviceToHost, S[2]));                                                                        \
    d_mem[2]._using = true;                                                                                                    \
}                                                                                                                            \
else {                                                                                                                        \
    checkCudaErrors(cudaMemcpy2DAsync(_dst->MatptrArr.ptr[_A->ArrayNumber - 1],                                                \
        _dst->pitch * sizeof(__type), d_mem[5].mem, pitch_B * sizeof(__type), _dst->width * sizeof(__type),                    \
        _dst->height, cudaMemcpyDeviceToHost, S[2]));                                                                        \
    d_mem[5]._using = true;                                                                                                    \
}                                                                                                                            \
                                                                                                                            \
}                                                                                                                            \





#define GEMM3_part_ABC(__type, _kernel_name) {                                                                                \
checkCudaErrors(cudaMemcpy2DAsync(d_mem[0].mem,                                                                                \
    pitch_A * sizeof(__type), _A->MatptrArr.ptr[0], _A->pitch * sizeof(__type),                                                \
    _A->width * sizeof(__type), _A->height, cudaMemcpyHostToDevice, S[1]));                                                    \
                                                                                                                            \
checkCudaErrors(cudaMemcpy2DAsync(d_mem[1].mem,                                                                                \
    pitch_B * sizeof(__type), _B->MatptrArr.ptr[0], _B->pitch * sizeof(__type),                                                \
    _B->width * sizeof(__type), _B->height, cudaMemcpyHostToDevice, S[1]));                                                    \
                                                                                                                            \
checkCudaErrors(cudaMemcpy2DAsync(d_mem[2].mem,                                                                                \
    pitch_B * sizeof(__type), _C->MatptrArr.ptr[0], _C->pitch * sizeof(__type),                                                \
    _C->width * sizeof(__type), _C->height, cudaMemcpyHostToDevice, S[1]));                                                    \
                                                                                                                            \
d_mem[0].leading = true;                                                                                                    \
d_mem[1].leading = true;                                                                                                    \
                                                                                                                            \
d_mem[3].leading = false;                                                                                                    \
d_mem[4].leading = false;                                                                                                    \
checkCudaErrors(cudaDeviceSynchronize());                                                                                    \
                                                                                                                            \
for (int i = 0; i < _A->ArrayNumber; ++i)                                                                                    \
{                                                                                                                            \
    if (i > 0) {                                                                                                            \
        if (d_mem[2].leading) {                                                                                                \
            checkCudaErrors(cudaMemcpy2DAsync(_dst->MatptrArr.ptr[i - 1],                                                    \
                _dst->pitch * sizeof(__type), d_mem[2].mem, pitch_B * sizeof(__type), _dst->width * sizeof(__type),            \
                _dst->height, cudaMemcpyDeviceToHost, S[2]));                                                                \
            d_mem[2]._using = true;                                                                                            \
        }                                                                                                                    \
        else {                                                                                                                \
            checkCudaErrors(cudaMemcpy2DAsync(_dst->MatptrArr.ptr[i - 1],                                                    \
                _dst->pitch * sizeof(__type), d_mem[5].mem, pitch_B * sizeof(__type), _dst->width * sizeof(__type),            \
                _dst->height, cudaMemcpyDeviceToHost, S[2]));                                                                \
            d_mem[5]._using = true;                                                                                            \
        }                                                                                                                    \
    }                                                                                                                        \
                                                                                                                            \
    if (d_mem[0].leading) {                                                                                                    \
        _kernel_name(d_mem[0].mem, d_mem[1].mem, d_mem[2].mem, d_mem[2].mem, pitch_A, pitch_B, hA, &S[0]);                    \
        d_mem[0]._using = true;                                                                                                \
        d_mem[1]._using = true;                                                                                                \
        d_mem[2]._using = true;                                                                                                \
                                                                                                                            \
        d_mem[2].leading = true;                                                                                            \
        d_mem[5].leading = false;                                                                                            \
    }                                                                                                                        \
    else {                                                                                                                    \
        _kernel_name(d_mem[3].mem, d_mem[4].mem, d_mem[5].mem, d_mem[5].mem, pitch_A, pitch_B, hA, &S[0]);                    \
        d_mem[3]._using = true;                                                                                                \
        d_mem[4]._using = true;                                                                                                \
        d_mem[5]._using = true;                                                                                                \
                                                                                                                            \
        d_mem[5].leading = true;                                                                                            \
        d_mem[2].leading = false;                                                                                            \
    }                                                                                                                        \
                                                                                                                            \
    if (i < _A->ArrayNumber - 1) {                                                                                            \
        if (!d_mem[0]._using) {                                                                                                \
            checkCudaErrors(cudaMemcpy2DAsync(d_mem[0].mem,                                                                    \
                pitch_A * sizeof(__type), _A->MatptrArr.ptr[i + 1], _A->pitch * sizeof(__type),                                \
                _A->width * sizeof(__type), _A->height, cudaMemcpyHostToDevice, S[1]));                                        \
                                                                                                                            \
            checkCudaErrors(cudaMemcpy2DAsync(d_mem[1].mem,                                                                    \
                pitch_B * sizeof(__type), _B->MatptrArr.ptr[i + 1], _B->pitch * sizeof(__type),                                \
                _B->width * sizeof(__type), _B->height, cudaMemcpyHostToDevice, S[1]));                                        \
                                                                                                                            \
            checkCudaErrors(cudaMemcpy2DAsync(d_mem[2].mem,                                                                    \
                pitch_B * sizeof(__type), _C->MatptrArr.ptr[i + 1], _C->pitch * sizeof(__type),                                \
                _C->width * sizeof(__type), _C->height, cudaMemcpyHostToDevice, S[1]));                                        \
                                                                                                                            \
            d_mem[0].leading = true;                                                                                        \
            d_mem[1].leading = true;                                                                                        \
                                                                                                                            \
            d_mem[3].leading = false;                                                                                        \
            d_mem[4].leading = false;                                                                                        \
        }                                                                                                                    \
        else {                                                                                                                \
            checkCudaErrors(cudaMemcpy2DAsync(d_mem[3].mem,                                                                    \
                pitch_A * sizeof(__type), _A->MatptrArr.ptr[i + 1], _A->pitch * sizeof(__type),                                \
                _A->width * sizeof(__type), _A->height, cudaMemcpyHostToDevice, S[1]));                                        \
                                                                                                                            \
            checkCudaErrors(cudaMemcpy2DAsync(d_mem[4].mem,                                                                    \
                pitch_B * sizeof(__type), _B->MatptrArr.ptr[i + 1], _B->pitch * sizeof(__type),                                \
                _B->width * sizeof(__type), _B->height, cudaMemcpyHostToDevice, S[1]));                                        \
                                                                                                                            \
            checkCudaErrors(cudaMemcpy2DAsync(d_mem[5].mem,                                                                    \
                pitch_B * sizeof(__type), _C->MatptrArr.ptr[i + 1], _C->pitch * sizeof(__type),                                \
                _C->width * sizeof(__type), _C->height, cudaMemcpyHostToDevice, S[1]));                                        \
                                                                                                                            \
            d_mem[3].leading = true;                                                                                        \
            d_mem[4].leading = true;                                                                                        \
                                                                                                                            \
            d_mem[0].leading = false;                                                                                        \
            d_mem[1].leading = false;                                                                                        \
        }                                                                                                                    \
    }                                                                                                                        \
                                                                                                                            \
    checkCudaErrors(cudaDeviceSynchronize());                                                                                \
                                                                                                                            \
    d_mem[0]._using = false;                                                                                                \
    d_mem[1]._using = false;                                                                                                \
    d_mem[2]._using = false;                                                                                                \
    d_mem[3]._using = false;                                                                                                \
    d_mem[4]._using = false;                                                                                                \
    d_mem[5]._using = false;                                                                                                \
}                                                                                                                            \
                                                                                                                            \
if (d_mem[2].leading) {                                                                                                        \
    checkCudaErrors(cudaMemcpy2DAsync(_dst->MatptrArr.ptr[_A->ArrayNumber - 1],                                                \
        _dst->pitch * sizeof(__type), d_mem[2].mem, pitch_B * sizeof(__type), _dst->width * sizeof(__type),                    \
        _dst->height, cudaMemcpyDeviceToHost, S[2]));                                                                        \
    d_mem[2]._using = true;                                                                                                    \
}                                                                                                                            \
else {                                                                                                                        \
    checkCudaErrors(cudaMemcpy2DAsync(_dst->MatptrArr.ptr[_A->ArrayNumber - 1],                                                \
        _dst->pitch * sizeof(__type), d_mem[5].mem, pitch_B * sizeof(__type), _dst->width * sizeof(__type),                    \
        _dst->height, cudaMemcpyDeviceToHost, S[2]));                                                                        \
    d_mem[5]._using = true;                                                                                                    \
}                                                                                                                            \
                                                                                                                            \
}                                                                                                                            \