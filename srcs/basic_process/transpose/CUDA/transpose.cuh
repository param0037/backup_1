/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _CUDA_TRANSPOSE_CUH_
#define _CUDA_TRANSPOSE_CUH_

#include "../../../core/basic.h"



#define TRANSPOSE_MAT4X4(_loc_transp, tmp)              \
{                                                       \
SWAP(_loc_transp[0].y, _loc_transp[1].x, tmp);          \
SWAP(_loc_transp[0].z, _loc_transp[2].x, tmp);          \
SWAP(_loc_transp[0].w, _loc_transp[3].x, tmp);          \
                                                        \
SWAP(_loc_transp[1].z, _loc_transp[2].y, tmp);          \
                                                        \
SWAP(_loc_transp[1].w, _loc_transp[3].y, tmp);          \
SWAP(_loc_transp[2].w, _loc_transp[3].z, tmp);          \
}


#define TRANSPOSE_MAT2X2(_loc_transp, tmp)              \
{                                                       \
SWAP(_loc_transp[0].y, _loc_transp[1].x, tmp);          \
}


__global__
/*
* @param width : In float4, dev_tmp->width / 4
* @param height : In float4, dev_tmp->height / 4
* @param true_dim_src : the true dimension of source matrix, measured in element
*/
void cu_transpose_vec4x4(float4* src, float4* dst, const uint width, const uint height, const uint2 true_dim_src)
{
#ifdef __CUDACC__
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;

    float4 _loc_transp[4];
    float tmp;
    size_t dex = 4 * (size_t)tidx * (size_t)width + (size_t)tidy;

    if (tidx < height && tidy < width) {
        if (4 * tidx < true_dim_src.y)          _loc_transp[0] = src[dex];
        dex += (size_t)width;
        if (4 * tidx + 1 < true_dim_src.y)      _loc_transp[1] = src[dex];
        dex += (size_t)width;
        if (4 * tidx + 2 < true_dim_src.y)      _loc_transp[2] = src[dex];
        dex += (size_t)width;
        if (4 * tidx + 3 < true_dim_src.y)      _loc_transp[3] = src[dex];
        dex += (size_t)width;

        TRANSPOSE_MAT4X4(_loc_transp, tmp);

        dex = 4 * (size_t)tidy * (size_t)height + (size_t)tidx;
        if (4 * tidy < true_dim_src.x)          dst[dex] = _loc_transp[0];
        dex += (size_t)height;
        if (4 * tidy + 1 < true_dim_src.x)      dst[dex] = _loc_transp[1];
        dex += (size_t)height;
        if (4 * tidy + 2 < true_dim_src.x)      dst[dex] = _loc_transp[2];
        dex += (size_t)height;
        if (4 * tidy + 3 < true_dim_src.x)      dst[dex] = _loc_transp[3];
        dex += (size_t)height;
    }
#endif
}




__global__
/*
* @param width : In float4, dev_tmp->width / 4
* @param height : In float4, dev_tmp->height / 4
* @param true_dim_src : the true dimension of source matrix, measured in element
*/
void cu_transpose_vec8x8(float4* src, float4* dst, const uint width, const uint height, const uint2 true_dim_src)
{
#ifdef __CUDACC__
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;

    float4 _loc_transp[8];
    // If I delete the volatile declaration, the entire register buffer will be set to zero by compiler
    volatile __half tmp;
    size_t dex = 8 * (size_t)tidx * (size_t)width + (size_t)tidy;

    if (tidx < height && tidy < width) {
        if (8 * tidx < true_dim_src.y)              _loc_transp[0] = src[dex];
        dex += (size_t)width;
        if (8 * tidx + 1 < true_dim_src.y)          _loc_transp[1] = src[dex];
        dex += (size_t)width;
        if (8 * tidx + 2 < true_dim_src.y)          _loc_transp[2] = src[dex];
        dex += (size_t)width;
        if (8 * tidx + 3 < true_dim_src.y)          _loc_transp[3] = src[dex];
        dex += (size_t)width;
        if (8 * tidx + 4 < true_dim_src.y)          _loc_transp[4] = src[dex];
        dex += (size_t)width;
        if (8 * tidx + 5 < true_dim_src.y)          _loc_transp[5] = src[dex];
        dex += (size_t)width;
        if (8 * tidx + 6 < true_dim_src.y)          _loc_transp[6] = src[dex];
        dex += (size_t)width;
        if (8 * tidx + 7 < true_dim_src.y)          _loc_transp[7] = src[dex];

        // OPS    -------------------------------------------------------------------------- | -------- Before Being Transposed ----------

        SWAP(*(((__half*)&_loc_transp[0]) + 1), *(((__half*)&_loc_transp[1])), tmp);                // row 0, col[1, 7]
        SWAP(*(((__half*)&_loc_transp[0]) + 2), *(((__half*)&_loc_transp[2])), tmp);
        SWAP(*(((__half*)&_loc_transp[0]) + 3), *(((__half*)&_loc_transp[3])), tmp);
        SWAP(*(((__half*)&_loc_transp[0]) + 4), *(((__half*)&_loc_transp[4])), tmp);
        SWAP(*(((__half*)&_loc_transp[0]) + 5), *(((__half*)&_loc_transp[5])), tmp);
        SWAP(*(((__half*)&_loc_transp[0]) + 6), *(((__half*)&_loc_transp[6])), tmp);
        SWAP(*(((__half*)&_loc_transp[0]) + 7), *(((__half*)&_loc_transp[7])), tmp);

        SWAP(*(((__half*)&_loc_transp[1]) + 2), *(((__half*)&_loc_transp[2]) + 1), tmp);            // row1, col[2, 7]
        SWAP(*(((__half*)&_loc_transp[1]) + 3), *(((__half*)&_loc_transp[3]) + 1), tmp);
        SWAP(*(((__half*)&_loc_transp[1]) + 4), *(((__half*)&_loc_transp[4]) + 1), tmp);
        SWAP(*(((__half*)&_loc_transp[1]) + 5), *(((__half*)&_loc_transp[5]) + 1), tmp);
        SWAP(*(((__half*)&_loc_transp[1]) + 6), *(((__half*)&_loc_transp[6]) + 1), tmp);
        SWAP(*(((__half*)&_loc_transp[1]) + 7), *(((__half*)&_loc_transp[7]) + 1), tmp);

        SWAP(*(((__half*)&_loc_transp[2]) + 3), *(((__half*)&_loc_transp[3]) + 2), tmp);            // row2, col[3, 7]
        SWAP(*(((__half*)&_loc_transp[2]) + 4), *(((__half*)&_loc_transp[4]) + 2), tmp);
        SWAP(*(((__half*)&_loc_transp[2]) + 5), *(((__half*)&_loc_transp[5]) + 2), tmp);
        SWAP(*(((__half*)&_loc_transp[2]) + 6), *(((__half*)&_loc_transp[6]) + 2), tmp);
        SWAP(*(((__half*)&_loc_transp[2]) + 7), *(((__half*)&_loc_transp[7]) + 2), tmp);

        SWAP(*(((__half*)&_loc_transp[3]) + 4), *(((__half*)&_loc_transp[4]) + 3), tmp);            // row3, col[4, 7]
        SWAP(*(((__half*)&_loc_transp[3]) + 5), *(((__half*)&_loc_transp[5]) + 3), tmp);
        SWAP(*(((__half*)&_loc_transp[3]) + 6), *(((__half*)&_loc_transp[6]) + 3), tmp);
        SWAP(*(((__half*)&_loc_transp[3]) + 7), *(((__half*)&_loc_transp[7]) + 3), tmp);

        SWAP(*(((__half*)&_loc_transp[4]) + 5), *(((__half*)&_loc_transp[5]) + 4), tmp);            // row4, col[5, 7]
        SWAP(*(((__half*)&_loc_transp[4]) + 6), *(((__half*)&_loc_transp[6]) + 4), tmp);
        SWAP(*(((__half*)&_loc_transp[4]) + 7), *(((__half*)&_loc_transp[7]) + 4), tmp);

        SWAP(*(((__half*)&_loc_transp[5]) + 6), *(((__half*)&_loc_transp[6]) + 5), tmp);            // row5, col[6, 7]
        SWAP(*(((__half*)&_loc_transp[5]) + 7), *(((__half*)&_loc_transp[7]) + 5), tmp);

        SWAP(*(((__half*)&_loc_transp[6]) + 7), *(((__half*)&_loc_transp[7]) + 6), tmp);            // row6, col[7, 7]

        dex = 8 * (size_t)tidy * (size_t)height + (size_t)tidx;
        if (8 * tidy < true_dim_src.x)              dst[dex] = _loc_transp[0];
        dex += (size_t)height;
        if (8 * tidy + 1 < true_dim_src.x)          dst[dex] = _loc_transp[1];
        dex += (size_t)height;
        if (8 * tidy + 2 < true_dim_src.x)          dst[dex] = _loc_transp[2];
        dex += (size_t)height;
        if (8 * tidy + 3 < true_dim_src.x)          dst[dex] = _loc_transp[3];
        dex += (size_t)height;
        if (8 * tidy + 4 < true_dim_src.x)          dst[dex] = _loc_transp[4];
        dex += (size_t)height;
        if (8 * tidy + 5 < true_dim_src.x)          dst[dex] = _loc_transp[5];
        dex += (size_t)height;
        if (8 * tidy + 6 < true_dim_src.x)          dst[dex] = _loc_transp[6];
        dex += (size_t)height;
        if (8 * tidy + 7 < true_dim_src.x)          dst[dex] = _loc_transp[7];
    }
#endif
}




__global__
/*
* @param width : In float4, dev_tmp->width / 4
* @param height : In float4, dev_tmp->height / 4
*/
void cu_transpose_vec2x2(double2* src, double2* dst, const uint width, const uint height, const uint2 true_dim_src)
{
#ifdef __CUDACC__
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;

    double2 _loc_transp[2];
    double tmp;
    size_t dex = 2 * (size_t)tidx * (size_t)width + (size_t)tidy;

    if (tidx < height && tidy < width) {
        if (2 * tidx < true_dim_src.y)              _loc_transp[0] = src[dex];
        dex += (size_t)width;
        if (2 * tidx + 1 < true_dim_src.y)          _loc_transp[1] = src[dex];

        TRANSPOSE_MAT2X2(_loc_transp, tmp);

        dex = 2 * (size_t)tidy * (size_t)height + (size_t)tidx;
        if (2 * tidy < true_dim_src.x)              dst[dex] = _loc_transp[0];
        dex += (size_t)height;
        if (2 * tidy + 1 < true_dim_src.x)          dst[dex] = _loc_transp[1];
    }
#endif
}


#endif
