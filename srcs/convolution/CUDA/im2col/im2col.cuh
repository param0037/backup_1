/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _IM2COL_CUH_
#define _IM2COL_CUH_

#include "../../../core/basic.h"



#define store_to_shmem_L_vec4 {                                                                            \
    frag[threadIdx.x][4 * threadIdx.y] = reg_0[0];                                                        \
    frag[threadIdx.x][4 * threadIdx.y + 1] = reg_0[1];                                                    \
    frag[threadIdx.x][4 * threadIdx.y + 2] = reg_0[2];                                                    \
    frag[threadIdx.x][4 * threadIdx.y + 3] = reg_0[3];                                                    \
    frag[16 + threadIdx.x][4 * threadIdx.y] = reg_1[0];                                                    \
    frag[16 + threadIdx.x][4 * threadIdx.y + 1] = reg_1[1];                                                \
    frag[16 + threadIdx.x][4 * threadIdx.y + 2] = reg_1[2];                                                \
    frag[16 + threadIdx.x][4 * threadIdx.y + 3] = reg_1[3];                                                \
}                                                                                                        \



#define store_to_shmem_R_vec4 {                                                                            \
    frag[threadIdx.x][64 + 4 * threadIdx.y] = reg_0[0];                                                    \
    frag[threadIdx.x][65 + 4 * threadIdx.y] = reg_0[1];                                                    \
    frag[threadIdx.x][66 + 4 * threadIdx.y] = reg_0[2];                                                    \
    frag[threadIdx.x][67 + 4 * threadIdx.y] = reg_0[3];                                                    \
    frag[16 + threadIdx.x][64 + 4 * threadIdx.y] = reg_1[0];                                            \
    frag[16 + threadIdx.x][65 + 4 * threadIdx.y] = reg_1[1];                                            \
    frag[16 + threadIdx.x][66 + 4 * threadIdx.y] = reg_1[2];                                            \
    frag[16 + threadIdx.x][67 + 4 * threadIdx.y] = reg_1[3];                                            \
}                                                



__device__ __inline__
void reg_left_shift_float4_4(float4* src)
{
    src[0] = src[1];
    src[1] = src[2];
    src[2] = src[3];
}


/*
* 因为要适应不同的边界模式，因此需要拷贝，那么就可以不用考虑线程的边界问题了, 但拷贝到dst还是要注意边界问题
*/


/**
* Block(16, 16), the radius of preset border is 8
*
*                     80 float4s     
*            --------------------------
*            |         apron              |        8 floats
*            |      --------------     |
*            |      |               |     |
*            |      | constant   |     |        32 floats  => __shared__ float src_frag[32][80]
*            |      |               |     |
*            |      --------------     |
*            |                         |
*            --------------------------
*/


__global__
/**
* Wdst = channel * kernel._element_num, 64x
* Hdst = src.element_num 16x
* Each thread process 4 float4s each loop, 4 lines in dst matrix
* @param thread_bound : the boundary of threads, .x : in float4, the width of src(the Tensor) / 4
*                                                 .y : the height of src(the Tensor)
* @param Wpitch_src : src_buf->Wpitch / 4, in float4
* @param depth_iter : how many loops should have along the depth, dpitch --> float4 once
* @param pitch_dst : the pitch of dst, which is equal to channel * kernel_width * kernel_height / 4, in float4
* @param ker_size : the size of kernel, (pitch, height)
*/
void cu_sIm2Col_r8_within(float4*                    src,
                          float4*                    dst,
                          const int2                kernel_shift,
                          const int2                thread_bound,
                          const size_t                Wpitch_src,
                          const size_t                pitch_dst,
                          const int2                ker_size,
                          const int                    depth)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    
    size_t glo_dex_src, glo_dex_dst;
    
    __shared__ float4 frag[32][80 + 1];
    float4 reg_0[4], 
        // [0] : shmem_ker_i, [1] : shmem_ker_j, [2] : ker_i, [3] : ker_j
           reg_1[4];
    
    for (int i = 0; i < depth; ++i)
    {
        glo_dex_src = (size_t)tidx * Wpitch_src + tidy * depth * 4 + i;

        reg_0[0] = src[glo_dex_src];
        reg_0[1] = src[glo_dex_src + depth];
        reg_0[2] = src[glo_dex_src + depth * 2];
        reg_0[3] = src[glo_dex_src + depth * 3];

        glo_dex_src += 16 * Wpitch_src;

        reg_1[0] = src[glo_dex_src];
        reg_1[1] = src[glo_dex_src + depth];
        reg_1[2] = src[glo_dex_src + depth * 2];
        reg_1[3] = src[glo_dex_src + depth * 3];

        store_to_shmem_L_vec4;

        if (threadIdx.y < 4) {
            glo_dex_src = (size_t)tidx * Wpitch_src + (tidy * 4 + 64) * depth + i;
            reg_0[0] = src[glo_dex_src];
            reg_0[1] = src[glo_dex_src + depth];
            reg_0[2] = src[glo_dex_src + depth * 2];
            reg_0[3] = src[glo_dex_src + depth * 3];

            glo_dex_src += 16 * Wpitch_src;

            reg_1[0] = src[glo_dex_src];
            reg_1[1] = src[glo_dex_src + depth];
            reg_1[2] = src[glo_dex_src + depth * 2];
            reg_1[3] = src[glo_dex_src + depth * 3];

            store_to_shmem_R_vec4;
        }

        __syncthreads();

        // glo_dex_src is as the height of dst
        glo_dex_src = ((size_t)tidx * thread_bound.x + (size_t)tidy) * 4;
        
        if (tidx < thread_bound.y && tidy < thread_bound.x) {
            for (int ker_iter = 0; ker_iter < ker_size.x * ker_size.y; ++ker_iter)
            {
                *((int*)&reg_1[2]) = ker_iter / ker_size.x;
                *((int*)&reg_1[3]) = ker_iter % ker_size.x;
                *((int*)&reg_1[0]) = kernel_shift.x + *((int*)&reg_1[2]);
                *((int*)&reg_1[1]) = kernel_shift.y + *((int*)&reg_1[3]);
                
                glo_dex_dst = glo_dex_src * pitch_dst + ker_iter * depth + i;
                
                if (*((int*)&reg_1[3])) {    // not the beginnig of each row on kernel
                    reg_left_shift_float4_4(reg_0);
                    reg_0[3] = frag[threadIdx.x + *((int*)&reg_1[0])][4 * threadIdx.y + *((int*)&reg_1[1]) + 3];
                }
                else {                            // the beginnig of each row on kernel
                    reg_0[0] = frag[threadIdx.x + *((int*)&reg_1[0])][4 * threadIdx.y + *((int*)&reg_1[1])];
                    reg_0[1] = frag[threadIdx.x + *((int*)&reg_1[0])][4 * threadIdx.y + *((int*)&reg_1[1]) + 1];
                    reg_0[2] = frag[threadIdx.x + *((int*)&reg_1[0])][4 * threadIdx.y + *((int*)&reg_1[1]) + 2];
                    reg_0[3] = frag[threadIdx.x + *((int*)&reg_1[0])][4 * threadIdx.y + *((int*)&reg_1[1]) + 3];
                }

                dst[glo_dex_dst] = reg_0[0];
                glo_dex_dst += pitch_dst;
                dst[glo_dex_dst] = reg_0[1];
                glo_dex_dst += pitch_dst;
                dst[glo_dex_dst] = reg_0[2];
                glo_dex_dst += pitch_dst;
                dst[glo_dex_dst] = reg_0[3];
            }
        }

        __syncthreads();
    }    // looping along the depth
}



/**
* Block(16, 16), the radius of preset border is 8
*
*                 66 float4s
*                                     1 floats
*            ------------------ ->apron 
*            | -------------- |
*            | |               | |
*            | |  constant  | |        18 floats  => __shared__ float src_frag[32][80]
*            | |               | |
*            | -------------- |
*            ------------------
*/


__global__
/**
* Wdst = channel * kernel._element_num, 64x
* Hdst = src.element_num 16x
* Each thread process 4 float4s each loop, 4 lines in dst matrix
* @param thread_bound : the boundary of threads, .x : in float4, the width of src(the Tensor) / 4
*                                                 .y : the height of src(the Tensor)
* @param Wpitch_src : src_buf->Wpitch / 4, in float4
* @param depth_iter : how many loops should have along the depth, dpitch --> float4 once
* @param pitch_dst : the pitch of dst, which is equal to channel * kernel_width * kernel_height / 4, in float4
* @param ker_size : the size of kernel, (pitch, height)
*/
void cu_sIm2Col_r1_within(float4*                    src,
                          float4*                    dst,
                          const int2                thread_bound,
                          const size_t                Wpitch_src,
                          const size_t                pitch_dst,
                          const int2                ker_size,
                          const int                    depth)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    
    size_t glo_dex_src, glo_dex_dst;
    // .x -. row, .y -> col
    int2 kernel_shift = make_int2(8 - ker_size.y / 2, 8 - ker_size.x / 2);

    __shared__ float4 frag[32][80 + 1];
    float4 reg_0[4], 
        // [0] : shmem_ker_i, [1] : shmem_ker_j, [2] : ker_i, [3] : ker_j
           reg_1[4];
    
    for (int i = 0; i < depth; ++i)
    {
        glo_dex_src = (size_t)tidx * Wpitch_src + tidy * depth * 4 + i;

        reg_0[0] = src[glo_dex_src];
        reg_0[1] = src[glo_dex_src + depth];
        reg_0[2] = src[glo_dex_src + depth * 2];
        reg_0[3] = src[glo_dex_src + depth * 3];

        glo_dex_src += 16 * Wpitch_src;

        reg_1[0] = src[glo_dex_src];
        reg_1[1] = src[glo_dex_src + depth];
        reg_1[2] = src[glo_dex_src + depth * 2];
        reg_1[3] = src[glo_dex_src + depth * 3];

        store_to_shmem_L_vec4;

        if (threadIdx.y < 4) {
            glo_dex_src = (size_t)tidx * Wpitch_src + (tidy * 4 + 16) * depth + i;
            reg_0[0] = src[glo_dex_src];
            reg_0[1] = src[glo_dex_src + depth];
            reg_0[2] = src[glo_dex_src + depth * 2];
            reg_0[3] = src[glo_dex_src + depth * 3];

            glo_dex_src += 16 * Wpitch_src;

            reg_1[0] = src[glo_dex_src];
            reg_1[1] = src[glo_dex_src + depth];
            reg_1[2] = src[glo_dex_src + depth * 2];
            reg_1[3] = src[glo_dex_src + depth * 3];

            store_to_shmem_R_vec4;
        }

        __syncthreads();

        // glo_dex_src is as the height of dst
        glo_dex_src = (size_t)tidx * thread_bound.x + (size_t)tidy * 4;

        if (tidx < thread_bound.y && tidy < thread_bound.x) {
            for (int ker_iter = 0; ker_iter < ker_size.x * ker_size.y; ++ker_iter)
            {
                *((int*)&reg_1[2]) = ker_iter / ker_size.x;
                *((int*)&reg_1[3]) = ker_iter % ker_size.x;
                *((int*)&reg_1[0]) = kernel_shift.x + *((int*)&reg_1[2]);
                *((int*)&reg_1[1]) = kernel_shift.y + *((int*)&reg_1[3]);
                
                glo_dex_dst = glo_dex_src * pitch_dst + i * (pitch_dst / depth) + ker_iter;

                if (*((int*)&reg_1[3])) {    // not the beginnig of each row on kernel
                    reg_left_shift_float4_4(reg_0);
                    reg_0[3] = reg_0[3] = frag[threadIdx.x + *((int*)&reg_1[0])][4 * threadIdx.y + *((int*)&reg_1[1]) + 3];
                }
                else {                            // the beginnig of each row on kernel
                    reg_0[0] = frag[threadIdx.x + *((int*)&reg_1[0])][4 * threadIdx.y + *((int*)&reg_1[1])];
                    reg_0[1] = frag[threadIdx.x + *((int*)&reg_1[0])][4 * threadIdx.y + *((int*)&reg_1[1]) + 1];
                    reg_0[2] = frag[threadIdx.x + *((int*)&reg_1[0])][4 * threadIdx.y + *((int*)&reg_1[1]) + 2];
                    reg_0[3] = frag[threadIdx.x + *((int*)&reg_1[0])][4 * threadIdx.y + *((int*)&reg_1[1]) + 3];
                }

                dst[glo_dex_dst] = reg_0[0];
                glo_dex_dst += pitch_dst;
                dst[glo_dex_dst] = reg_0[1];
                glo_dex_dst += pitch_dst;
                dst[glo_dex_dst] = reg_0[2];
                glo_dex_dst += pitch_dst;
                dst[glo_dex_dst] = reg_0[3];
            }
        }
    }    // looping along the depth
}




#endif