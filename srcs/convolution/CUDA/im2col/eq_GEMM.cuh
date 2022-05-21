/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright © Wayne,
*    2021.04.16
*/


#ifndef _EQ_GEMM_CUH_
#define _EQ_GEMM_CUH_

/*
* In the equivalent GEMM kernel, whether the frag_B(name of the shared memory used to load from matrix B)
* is transposed. In fact, Nvidia© Nsight System proofs that the two different kernels(transposed or not)
* perform almost the same.
*/
#define _FRAG_B_TRANSPOSED_

#include "../../../core/basic.h"


#define _SET_ZERO_FLOAT4_1x4_(arr){                    \
    arr[0] = make_float4(0, 0, 0, 0);                \
    arr[1] = make_float4(0, 0, 0, 0);                \
    arr[2] = make_float4(0, 0, 0, 0);                \
    arr[3] = make_float4(0, 0, 0, 0);                \
}




#define _SET_ZERO_FLOAT4_1x8_(arr){                    \
    arr[0] = make_float4(0, 0, 0, 0);                \
    arr[1] = make_float4(0, 0, 0, 0);                \
    arr[2] = make_float4(0, 0, 0, 0);                \
    arr[3] = make_float4(0, 0, 0, 0);                \
    arr[4] = make_float4(0, 0, 0, 0);                \
    arr[5] = make_float4(0, 0, 0, 0);                \
    arr[6] = make_float4(0, 0, 0, 0);                \
    arr[7] = make_float4(0, 0, 0, 0);                \
}



#define _SET_ZERO_FLOAT4_2x4_(arr){                    \
    arr[0][0] = make_float4(0, 0, 0, 0);            \
    arr[1][0] = make_float4(0, 0, 0, 0);            \
    arr[2][0] = make_float4(0, 0, 0, 0);            \
    arr[3][0] = make_float4(0, 0, 0, 0);            \
    arr[0][1] = make_float4(0, 0, 0, 0);            \
    arr[1][1] = make_float4(0, 0, 0, 0);            \
    arr[2][1] = make_float4(0, 0, 0, 0);            \
    arr[3][1] = make_float4(0, 0, 0, 0);            \
}


#define _STORE_TO_DEST_(reg_sum_dex){    dst[dex_A] = reg_sum[reg_sum_dex];        dex_A += dst_store_bound; }


#ifdef _FRAG_B_TRANSPOSED_


#define _MM_ONE_ELEMENT_(sum_label, A_shift, B_shift){                                                                \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][0].x, reg_B[2 * B_shift].x, reg_sum[A_shift].sum_label);        \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][0].y, reg_B[2 * B_shift].y, reg_sum[A_shift].sum_label);        \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][0].z, reg_B[2 * B_shift].z, reg_sum[A_shift].sum_label);        \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][0].w, reg_B[2 * B_shift].w, reg_sum[A_shift].sum_label);        \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][1].x, reg_B[2 * B_shift + 1].x, reg_sum[A_shift].sum_label);    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][1].y, reg_B[2 * B_shift + 1].y, reg_sum[A_shift].sum_label);    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][1].z, reg_B[2 * B_shift + 1].z, reg_sum[A_shift].sum_label);    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][1].w, reg_B[2 * B_shift + 1].w, reg_sum[A_shift].sum_label);    \
}


#define _MM_4x8_ONE_ROW_(row) {                        \
_MM_ONE_ELEMENT_(x, row, 0);                        \
_MM_ONE_ELEMENT_(y, row, 1);                        \
_MM_ONE_ELEMENT_(z, row, 2);                        \
_MM_ONE_ELEMENT_(w, row, 3);                        \
}



#define _MM_4x4x8_ {            \
_MM_4x8_ONE_ROW_(0);            \
_MM_4x8_ONE_ROW_(1);            \
_MM_4x8_ONE_ROW_(2);            \
_MM_4x8_ONE_ROW_(3);            \
}





__global__
/*
* Threads distribution : int(8 x 32) = (256)
* Matrix A has limitation on its boundaries, but Matrix B dose not
* 
* @param MatA_load_bounds : .x : width of matrix A(in float4) (how many float4 in a row)
*                            .y : height of matrix A(in float4), I2C->height is 4x (how many float4 in a colume)
* @param dst_store_bound : the width of dst, in float4
* @param WB : width of matrix B (in float4)
* @param HB : height of kernel_buf (in float)
* @param __iter : how many iteration along B's width (128x)
*/
void cu_conv_eq_mm(float4*                A,
                   float4*                B,
                   float4*                dst,
                   const ulong2            MatA_load_bounds,
                   const uint            dst_store_bound,            // in float4
                   const uint            WB,                            // in float4
                   const uint            __iter)                        // how many iteration along B's width (128x)
{
    const uint tidx = threadIdx.x + blockIdx.x * blockDim.x,    // [0, 32)
               tidy = threadIdx.y + blockIdx.y * blockDim.y;    // [0, 8)

    uint along_L_A;

    float4 reg_A[4][2], reg_B[8], reg_sum[4];

    _SET_ZERO_FLOAT4_1x4_(reg_sum);

    __shared__ float4 frag_A[128][8 + 1];
    __shared__ float4 frag_B[32][32 + 1];

    size_t dex_A, dex_B, dex_dst;

    along_L_A = threadIdx.y;
    dex_A = tidx * MatA_load_bounds.x * 4 + along_L_A;
    dex_B = tidy * WB * 4 + threadIdx.x;

    for (uint i = 0; i < __iter; ++i)
    {
        _SET_ZERO_FLOAT4_1x8_(reg_B);

        reg_B[0] = B[dex_B];
        reg_B[1] = B[dex_B + WB];
        reg_B[2] = B[dex_B + 2 * WB];
        reg_B[3] = B[dex_B + 3 * WB];
        

        frag_B[threadIdx.y * 4][threadIdx.x] = reg_B[0];
        frag_B[threadIdx.y * 4 + 1][threadIdx.x] = reg_B[1];
        frag_B[threadIdx.y * 4 + 2][threadIdx.x] = reg_B[2];
        frag_B[threadIdx.y * 4 + 3][threadIdx.x] = reg_B[3];

        for (uint load_A = 0; load_A < 4; ++load_A) {
            // square 00
            _SET_ZERO_FLOAT4_2x4_(reg_A);

            if (tidx < MatA_load_bounds.y && along_L_A < MatA_load_bounds.x) {
                reg_A[0][0] = A[dex_A];
                reg_A[1][0] = A[dex_A + MatA_load_bounds.x];
                reg_A[2][0] = A[dex_A + MatA_load_bounds.x * 2];
                reg_A[3][0] = A[dex_A + MatA_load_bounds.x * 3];
            }

            frag_A[threadIdx.x * 4][threadIdx.y] = reg_A[0][0];
            frag_A[threadIdx.x * 4 + 1][threadIdx.y] = reg_A[1][0];
            frag_A[threadIdx.x * 4 + 2][threadIdx.y] = reg_A[2][0];
            frag_A[threadIdx.x * 4 + 3][threadIdx.y] = reg_A[3][0];

            along_L_A += 8;
            dex_A += 8;

            __syncthreads();

            // calculation
            for (uint _L = 0; _L < 4; ++_L) {
                
                reg_A[0][0] = frag_A[threadIdx.x * 4][_L * 2];
                reg_A[0][1] = frag_A[threadIdx.x * 4][_L * 2 + 1];
                reg_A[1][0] = frag_A[threadIdx.x * 4 + 1][_L * 2];
                reg_A[1][1] = frag_A[threadIdx.x * 4 + 1][_L * 2 + 1];
                reg_A[2][0] = frag_A[threadIdx.x * 4 + 2][_L * 2];
                reg_A[2][1] = frag_A[threadIdx.x * 4 + 2][_L * 2 + 1];
                reg_A[3][0] = frag_A[threadIdx.x * 4 + 3][_L * 2];
                reg_A[3][1] = frag_A[threadIdx.x * 4 + 3][_L * 2 + 1];

                reg_B[0] = frag_B[threadIdx.y * 4][load_A * 8 + _L * 2];
                reg_B[1] = frag_B[threadIdx.y * 4][load_A * 8 + _L * 2 + 1];
                reg_B[2] = frag_B[threadIdx.y * 4 + 1][load_A * 8 + _L * 2];
                reg_B[3] = frag_B[threadIdx.y * 4 + 1][load_A * 8 + _L * 2 + 1];
                reg_B[4] = frag_B[threadIdx.y * 4 + 2][load_A * 8 + _L * 2];
                reg_B[5] = frag_B[threadIdx.y * 4 + 2][load_A * 8 + _L * 2 + 1];
                reg_B[6] = frag_B[threadIdx.y * 4 + 3][load_A * 8 + _L * 2];
                reg_B[7] = frag_B[threadIdx.y * 4 + 3][load_A * 8 + _L * 2 + 1];

                _MM_4x4x8_;
            }

            __syncthreads();
        }

        dex_B += 32;
    }

    dex_A = tidx * dst_store_bound * 4 + tidy;

    if (tidy < dst_store_bound) {
        _STORE_TO_DEST_(0);
        _STORE_TO_DEST_(1);
        _STORE_TO_DEST_(2);
        _STORE_TO_DEST_(3);
    }
}


#else


#define _MM_ONE_ELEMENT_(sum_label, A_shift, B_label){                                                                        \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][0].x, reg_B[0].B_label, reg_sum[A_shift].sum_label);                    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][0].y, reg_B[1].B_label, reg_sum[A_shift].sum_label);                    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][0].z, reg_B[2].B_label, reg_sum[A_shift].sum_label);                    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][0].w, reg_B[3].B_label, reg_sum[A_shift].sum_label);                    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][1].x, reg_B[4].B_label, reg_sum[A_shift].sum_label);                    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][1].y, reg_B[5].B_label, reg_sum[A_shift].sum_label);                    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][1].z, reg_B[6].B_label, reg_sum[A_shift].sum_label);                    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][1].w, reg_B[7].B_label, reg_sum[A_shift].sum_label);                    \
}


#define _MM_4x8_ONE_ROW_(row) {                            \
    _MM_ONE_ELEMENT_(x, row, x);                        \
    _MM_ONE_ELEMENT_(y, row, y);                        \
    _MM_ONE_ELEMENT_(z, row, z);                        \
    _MM_ONE_ELEMENT_(w, row, w);                        \
}


#define _MM_4x4x8_ {                \
    _MM_4x8_ONE_ROW_(0);            \
    _MM_4x8_ONE_ROW_(1);            \
    _MM_4x8_ONE_ROW_(2);            \
    _MM_4x8_ONE_ROW_(3);            \
}




__global__
/*
* Threads distribution : int(8 x 32) = (256)
* Matrix A has limitation on its boundaries, but Matrix B dose not
*
* @param MatA_load_bounds : .x : width of matrix A(in float4) (how many float4 in a row)
*                            .y : height of matrix A(in float4), I2C->height is 4x (how many float4 in a colume)
* @param WB : width of matrix B (in float4)
* @param HB : height of kernel_buf (in float)
* @param __iter : how many iteration along B's width (128x)
*/
void cu_conv_eq_mm(float4*                A,
                   float4*                B,
                   float4*                dst,
                   const ulong2            MatA_load_bounds,
                   const uint            dst_store_bound,            // in float4
                   const uint            WB,                            // in float4
                   const uint            __iter)                        // how many iteration along B's width (128x)
{
    const uint tidx = threadIdx.x + blockIdx.x * blockDim.x,
        tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint along_L_A;

    float4 reg_A[4][2], reg_B[8], reg_sum[4];

    _SET_ZERO_FLOAT4_1x4_(reg_sum);

    __shared__ float4 frag_A[128][8 + 1];
    __shared__ float4 frag_B[128][8 + 1];

    size_t dex_A, dex_B, dex_dst;

    along_L_A = threadIdx.y;
    dex_A = tidx * MatA_load_bounds.x * 4 + along_L_A;
    dex_B = tidy * WB * 4 + threadIdx.x;

    for (uint i = 0; i < __iter; ++i)
    {
        _SET_ZERO_FLOAT4_1x8_(reg_B);

        reg_B[0] = B[dex_B];
        reg_B[1] = B[dex_B + WB];
        reg_B[2] = B[dex_B + 2 * WB];
        reg_B[3] = B[dex_B + 3 * WB];

        // transpose
        reg_B[4].x = reg_B[0].x;    reg_B[4].y = reg_B[1].x;    reg_B[4].z = reg_B[2].x;    reg_B[4].w = reg_B[3].x;
        reg_B[5].x = reg_B[0].y;    reg_B[5].y = reg_B[1].y;    reg_B[5].z = reg_B[2].y;    reg_B[5].w = reg_B[3].y;
        reg_B[6].x = reg_B[0].z;    reg_B[6].y = reg_B[1].z;    reg_B[6].z = reg_B[2].z;    reg_B[6].w = reg_B[3].z;
        reg_B[7].x = reg_B[0].w;    reg_B[7].y = reg_B[1].w;    reg_B[7].z = reg_B[2].w;    reg_B[7].w = reg_B[3].w;

        frag_B[threadIdx.x * 4][threadIdx.y] = reg_B[4];
        frag_B[threadIdx.x * 4 + 1][threadIdx.y] = reg_B[5];
        frag_B[threadIdx.x * 4 + 2][threadIdx.y] = reg_B[6];
        frag_B[threadIdx.x * 4 + 3][threadIdx.y] = reg_B[7];

        for (uint load_A = 0; load_A < 4; ++load_A) {
            // square 00
            _SET_ZERO_FLOAT4_2x4_(reg_A);

            if (tidx < MatA_load_bounds.y && along_L_A < MatA_load_bounds.x) {
                reg_A[0][0] = A[dex_A];
                reg_A[1][0] = A[dex_A + MatA_load_bounds.x];
                reg_A[2][0] = A[dex_A + MatA_load_bounds.x * 2];
                reg_A[3][0] = A[dex_A + MatA_load_bounds.x * 3];
            }

            frag_A[threadIdx.x * 4][threadIdx.y] = reg_A[0][0];
            frag_A[threadIdx.x * 4 + 1][threadIdx.y] = reg_A[1][0];
            frag_A[threadIdx.x * 4 + 2][threadIdx.y] = reg_A[2][0];
            frag_A[threadIdx.x * 4 + 3][threadIdx.y] = reg_A[3][0];

            along_L_A += 8;
            dex_A += 8;

            __syncthreads();

            // calculation
            for (uint _L = 0; _L < 4; ++_L) {

                reg_A[0][0] = frag_A[threadIdx.x * 4][_L * 2];
                reg_A[0][1] = frag_A[threadIdx.x * 4][_L * 2 + 1];
                reg_A[1][0] = frag_A[threadIdx.x * 4 + 1][_L * 2];
                reg_A[1][1] = frag_A[threadIdx.x * 4 + 1][_L * 2 + 1];
                reg_A[2][0] = frag_A[threadIdx.x * 4 + 2][_L * 2];
                reg_A[2][1] = frag_A[threadIdx.x * 4 + 2][_L * 2 + 1];
                reg_A[3][0] = frag_A[threadIdx.x * 4 + 3][_L * 2];
                reg_A[3][1] = frag_A[threadIdx.x * 4 + 3][_L * 2 + 1];

                reg_B[0] = frag_B[load_A * 32 + _L * 8 + 0][threadIdx.y];
                reg_B[1] = frag_B[load_A * 32 + _L * 8 + 1][threadIdx.y];
                reg_B[2] = frag_B[load_A * 32 + _L * 8 + 2][threadIdx.y];
                reg_B[3] = frag_B[load_A * 32 + _L * 8 + 3][threadIdx.y];
                reg_B[4] = frag_B[load_A * 32 + _L * 8 + 4][threadIdx.y];
                reg_B[5] = frag_B[load_A * 32 + _L * 8 + 5][threadIdx.y];
                reg_B[6] = frag_B[load_A * 32 + _L * 8 + 6][threadIdx.y];
                reg_B[7] = frag_B[load_A * 32 + _L * 8 + 7][threadIdx.y];

                _MM_4x4x8_;
            }

            __syncthreads();
        }

        dex_B += 32;
    }

    dex_A = tidx * dst_store_bound * 4 + tidy;

    if (tidy < dst_store_bound){
        _STORE_TO_DEST_(0);
        _STORE_TO_DEST_(1);
        _STORE_TO_DEST_(2);
        _STORE_TO_DEST_(3);
    }
}


#endif        // #ifdef _FRAG_B_TRANSPOSED_

#endif