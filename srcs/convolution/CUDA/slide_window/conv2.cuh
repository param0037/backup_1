/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _CONV2_CUH_
#define _CONV2_CUH_

#include "../../../core/basic.h"
#include "../../../classes/classes_util.h"

#define init_valuef 0.f
#define init_valueUint 0



__device__ __inline
void reg_shift_fp16(half2_8* tmp_reg_ptr)
{
#if __ABOVE_SM_53
#pragma unroll 7
    for (int i = 1; i < 8; ++i) {
        __half tmp = ((__half*)tmp_reg_ptr)[i];
        ((__half*)tmp_reg_ptr)[i - 1] = tmp;
    }
#endif
}



__device__ __inline
void reg_shift_f(float4* tmp_reg_ptr)
{
#pragma unroll 3
    for (int i = 1; i < 4; ++i) {
        float tmp = ((float*)tmp_reg_ptr)[i];
        ((float*)tmp_reg_ptr)[i - 1] = tmp;
    }
}



/**
* 传统二维滑窗卷积, traditional 2D convolution by shifting kernel
* 归并，将不同大小的卷积核离散到几个特定的卷积核大小优化核上进行
* 
* /////////////////////////////////////////////////////////////////////////////////////
*/


#define store_to_shmem_L {                                                                                    \
    src_frag[threadIdx.x][4 * threadIdx.y] = reg_0.x;                                                        \
    src_frag[threadIdx.x][4 * threadIdx.y + 1] = reg_0.y;                                                    \
    src_frag[threadIdx.x][4 * threadIdx.y + 2] = reg_0.z;                                                    \
    src_frag[threadIdx.x][4 * threadIdx.y + 3] = reg_0.w;                                                    \
    src_frag[16 + threadIdx.x][4 * threadIdx.y] = reg_1.x;                                                    \
    src_frag[16 + threadIdx.x][4 * threadIdx.y + 1] = reg_1.y;                                                \
    src_frag[16 + threadIdx.x][4 * threadIdx.y + 2] = reg_1.z;                                                \
    src_frag[16 + threadIdx.x][4 * threadIdx.y + 3] = reg_1.w;                                                \
}                                                                                                            \


#define hstore_to_shmem_L {                                                                                    \
    *((float*)&src_frag[threadIdx.x][8 * threadIdx.y]) = ((float4*)&reg_0)->x;                                \
    *((float*)&src_frag[threadIdx.x][8 * threadIdx.y + 2]) = ((float4*)&reg_0)->y;                            \
    *((float*)&src_frag[threadIdx.x][8 * threadIdx.y + 4]) = ((float4*)&reg_0)->z;                            \
    *((float*)&src_frag[threadIdx.x][8 * threadIdx.y + 6]) = ((float4*)&reg_0)->w;                            \
    *((float*)&src_frag[16 + threadIdx.x][8 * threadIdx.y]) = ((float4*)&reg_1)->x;                            \
    *((float*)&src_frag[16 + threadIdx.x][8 * threadIdx.y + 2]) = ((float4*)&reg_1)->y;                        \
    *((float*)&src_frag[16 + threadIdx.x][8 * threadIdx.y + 4]) = ((float4*)&reg_1)->z;                        \
    *((float*)&src_frag[16 + threadIdx.x][8 * threadIdx.y + 6]) = ((float4*)&reg_1)->w;                        \
}                                                                                                            \



#define store_to_shmem_R {                                                                                    \
    src_frag[threadIdx.x][64 + 4 * threadIdx.y] = reg_0.x;                                                    \
    src_frag[threadIdx.x][65 + 4 * threadIdx.y] = reg_0.y;                                                    \
    src_frag[threadIdx.x][66 + 4 * threadIdx.y] = reg_0.z;                                                    \
    src_frag[threadIdx.x][67 + 4 * threadIdx.y] = reg_0.w;                                                    \
    src_frag[16 + threadIdx.x][64 + 4 * threadIdx.y] = reg_1.x;                                                \
    src_frag[16 + threadIdx.x][65 + 4 * threadIdx.y] = reg_1.y;                                                \
    src_frag[16 + threadIdx.x][66 + 4 * threadIdx.y] = reg_1.z;                                                \
    src_frag[16 + threadIdx.x][67 + 4 * threadIdx.y] = reg_1.w;                                                \
}                                                                                                            \


#define hstore_to_shmem_R {    \
    *((float*)&src_frag[threadIdx.x][128 + 8 * threadIdx.y]) = ((float4*)&reg_0)->x;                        \
    *((float*)&src_frag[threadIdx.x][130 + 8 * threadIdx.y]) = ((float4*)&reg_0)->y;                        \
    *((float*)&src_frag[threadIdx.x][132 + 8 * threadIdx.y]) = ((float4*)&reg_0)->z;                        \
    *((float*)&src_frag[threadIdx.x][134 + 8 * threadIdx.y]) = ((float4*)&reg_0)->w;                        \
    *((float*)&src_frag[16 + threadIdx.x][128 + 8 * threadIdx.y]) = ((float4*)&reg_1)->x;                    \
    *((float*)&src_frag[16 + threadIdx.x][130 + 8 * threadIdx.y]) = ((float4*)&reg_1)->y;                    \
    *((float*)&src_frag[16 + threadIdx.x][132 + 8 * threadIdx.y]) = ((float4*)&reg_1)->z;                    \
    *((float*)&src_frag[16 + threadIdx.x][134 + 8 * threadIdx.y]) = ((float4*)&reg_1)->w;                    \
}                                                                                                            \



#define Conv_fmaf {                                                                                            \
    reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);                                                                \
    reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);                                                                \
    reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);                                                                \
    reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);                                                                \
}                                                                                                            \



#define hstore_to_shmem_L3(offset_x) {                                                                        \
    *((float*)&src_frag[offset_x + threadIdx.x][8 * threadIdx.y]) = ((float4*)&reg_0)->x;                    \
    *((float*)&src_frag[offset_x + threadIdx.x][8 * threadIdx.y + 2]) = ((float4*)&reg_0)->y;                \
    *((float*)&src_frag[offset_x + threadIdx.x][8 * threadIdx.y + 4]) = ((float4*)&reg_0)->z;                \
    *((float*)&src_frag[offset_x + threadIdx.x][8 * threadIdx.y + 6]) = ((float4*)&reg_0)->w;                \
}                                                                                                            \



#define store_to_shmem_L3(offset_x) {                                                                        \
    src_frag[offset_x + threadIdx.x][4 * threadIdx.y] = reg_0.x;                                            \
    src_frag[offset_x + threadIdx.x][4 * threadIdx.y + 1] = reg_0.y;                                        \
    src_frag[offset_x + threadIdx.x][4 * threadIdx.y + 2] = reg_0.z;                                        \
    src_frag[offset_x + threadIdx.x][4 * threadIdx.y + 3] = reg_0.w;                                        \
}                                                                                                            \



#define store_to_shmem_R3(offset_x) {                                                                        \
    src_frag[offset_x + threadIdx.x][64 + 4 * threadIdx.y] = reg_0.x;                                        \
    src_frag[offset_x + threadIdx.x][65 + 4 * threadIdx.y] = reg_0.y;                                        \
    src_frag[offset_x + threadIdx.x][66 + 4 * threadIdx.y] = reg_0.z;                                        \
    src_frag[offset_x + threadIdx.x][67 + 4 * threadIdx.y] = reg_0.w;                                        \
}                                                                                                            \



#define hstore_to_shmem_R3(offset_x) {                                                                        \
    *((float*)&src_frag[offset_x + threadIdx.x][128 + 8 * threadIdx.y]) = ((float4*)&reg_0)->x;                \
    *((float*)&src_frag[offset_x + threadIdx.x][130 + 8 * threadIdx.y]) = ((float4*)&reg_0)->y;                \
    *((float*)&src_frag[offset_x + threadIdx.x][132 + 8 * threadIdx.y]) = ((float4*)&reg_0)->z;                \
    *((float*)&src_frag[offset_x + threadIdx.x][134 + 8 * threadIdx.y]) = ((float4*)&reg_0)->w;                \
}                                                                                                            \



#define sharedmem_offset 1        // to mitigate even prevent blank conflict


// [s][h]Conv2_[exact][within]_[offset]

// [s]        -> single precision floating point (fp32)
// [h]        -> half precision floating point (fp16)
// [exact]    -> The kernel size is exactly fitting the loading size
// [exact]    -> The kernel size is within the loading size
// [offset] -> especially for the convolution with multi channels, indicating the offset of kernel data on constant memory,
//        in the stride of sizeof corresponding type.

// ---------------------------------------- 8 x 8 -------------------------------------------------------------

extern "C"
{


__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem[32][80]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions
* 
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
* 
*             \80 floats     8 floats
* ----------------------------------
* |                                |        8 halfs
* |        -----------------       |
* |       |                 |      |
* |  apron|     constant    |      |        32 floats  => __shared__ float src_frag[32][80]
* |       |                 |      |
* |        -----------------       |
* |                                |
* ----------------------------------
*/
void cu_sConv2_r8_exact(float4*            src, 
                        float4*            dst,
                        const uint        pitch_src, 
                        const uint        pitch_dst,
                        const uint        total_ker_len, 
                        const uint        Wker)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[32][80 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    glo_dex += 16 * pitch_src;
    reg_1 = src[glo_dex];

    store_to_shmem_L
    
    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        glo_dex += 16 * pitch_src;
        reg_1 = src[glo_dex];

        store_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];

        }
        tmp_ker = ((float*)Const_Mem)[i];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}





__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x8个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 8 + 8 * 2)*(16 + 8 * 2) 即shmem half[32][144] -> float[32][72]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions
*
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
*
*         \144halfs(72 floats)     8 halfs
* ----------------------------------
* |                                |        8 halfs
* |        -----------------       |
* |       |                 |      |
* |  apron|     constant    |      |        
* |       |                 |      |
* |        -----------------       |
* |                                |
* ----------------------------------
*/
void cu_hConv2_r8_exact(float4*            src,
                        float4*            dst,
                        const uint        pitch_src,
                        const uint        pitch_dst,
                        const uint        total_ker_len,
                        const uint        Wker)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[32][144 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    glo_dex += 16 * pitch_src;
    *((float4*)&reg_1) = src[glo_dex];

    hstore_to_shmem_L
    
    if (threadIdx.y < 2) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        glo_dex += 16 * pitch_src;
        *((float4*)&reg_1) = src[glo_dex];

        hstore_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = ((__half*)Const_Mem)[i];
        tmp_ker.y = tmp_ker.x;
        
        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&reg_1);
#endif
}


__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem[32][80]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions(if float4 is consider horizentally, then +4 at width)
*
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度在8以内
* */
void cu_sConv2_r8_within(float4*            src, 
                         float4*            dst,
                         const uint            pitch_src, 
                         const uint            pitch_dst,
                         const uint            total_ker_len, 
                         const uint            Wker,
                         const int2            kernel_shift)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[32][80 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    glo_dex += 16 * pitch_src;
    reg_1 = src[glo_dex];

    store_to_shmem_L
    
    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        glo_dex += 16 * pitch_src;
        reg_1 = src[glo_dex];

        store_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)Const_Mem)[i];
        
        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;
}



__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x8个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 8 + 8 * 2)*(16 + 8 * 2) 即shmem half[32][144] -> float[32][72]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions
*
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
*
*         \144halfs(72 floats)     8 halfs
* -------------------------------
* |                                |        8 halfs
* |         -----------------        |
* |         |                 |        |
* |    apron|     constant     |        |        
* |         |                 |        |
* |         -----------------        |
* |                                |
* -------------------------------
*/
void cu_hConv2_r8_within(float4*            src, 
                         float4*            dst,
                         const uint            pitch_src, 
                         const uint            pitch_dst,
                         const uint            total_ker_len, 
                         const uint            Wker,
                         const int2            kernel_shift)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[32][144 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    glo_dex += 16 * pitch_src;
    *((float4*)&reg_1) = src[glo_dex];

    hstore_to_shmem_L
    
    if (threadIdx.y < 2) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        glo_dex += 16 * pitch_src;
        *((float4*)&reg_1) = src[glo_dex];

        hstore_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = ((__half*)Const_Mem)[i];
        tmp_ker.y = tmp_ker.x;
        
        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    /*for (int i = 0; i < 8; ++i) {
        ((__half*)&reg_1)[i] = 255.f;
    }*/
    dst[glo_dex] = *((float4*)&reg_1);
#endif
}



}



// ---------------------------------------- 16 x 16 -------------------------------------------------------------




extern "C"
{
__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem float[48][96]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 16 * 2 = 32 on all directions(if float4 is consider horizentally, then +8 at width)
* 
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
* 
*            \96 floats     16 floats
* -----------------------------------
* |                                 |        16 floats
* |         -----------------       |
* |         |               |       |
* |    apron|     constant  |       |        48 floats  => __shared__ float src_frag[32][80]
* |         |               |       |
* |         -----------------       |
* |                                 |
* -----------------------------------
*/
void cu_sConv2_r16_exact(float4*               src, 
                         float4*               dst,
                         const uint            pitch_src, 
                         const uint            pitch_dst,
                         const uint            total_ker_len, 
                         const uint            Wker)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[48][96 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(32)

    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)Const_Mem)[i];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}



__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 128 + 16 * 2)*(16 + 16 * 2) 即shmem half[48][160]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 16 * 2 = 32 on all directions(if float4 is consider horizentally, then +8 at width)
* 
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
* 
*     \160 floats (80 floats)    16 floats (20 float4s = (16 + 4) float4s)
* -------------------------------
* |                                |        16 halfs
* |         -----------------        |
* |         |                 |        |
* |    apron|     constant     |        |        48 halfs  => __shared__ float src_frag[48][160]
* |         |                 |        |
* |         -----------------        |
* |                                |
* -------------------------------
*/
void cu_hConv2_r16_exact(float4*            src, 
                         float4*            dst,
                         const uint            pitch_src, 
                         const uint            pitch_dst,
                         const uint            total_ker_len, 
                         const uint            Wker)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[48][160 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(32)

    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = ((__half*)Const_Mem)[i];
        tmp_ker.y = tmp_ker.x;

        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&reg_1);
#endif
}



__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem[32][80]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions(if float4 is consider horizentally, then +4 at width)
*
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度在8以内
* */
void cu_sConv2_r16_within(float4*            src, 
                          float4*            dst,
                          const uint        pitch_src, 
                          const uint        pitch_dst,
                          const uint        total_ker_len, 
                          const uint        Wker,
                          const int2        kernel_shift)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float src_frag[48][96 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(32)

    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(0)

            glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(16)

            glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)Const_Mem)[i];
        
        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}


__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem[32][80]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions(if float4 is consider horizentally, then +4 at width)
*
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度在8以内
* */
void cu_hConv2_r16_within(float4*                src, 
                          float4*                dst,
                          const uint            pitch_src, 
                          const uint            pitch_dst,
                          const uint            total_ker_len, 
                          const uint            Wker,
                          const int2            kernel_shift)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[48][160 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(32)

    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = ((__half*)Const_Mem)[i];
        tmp_ker.y = tmp_ker.x;

        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&reg_1);
#endif
}


}




// ---------------------------------------- 8 x 16 -------------------------------------------------------------


extern "C"
{
__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem float[32][96]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dimension of Dsrc is +16 * 2 = 32floats (+8 float4s if considered float4) on width
* The dimension of Dsrc is +8 * 2 = 16floats on height
* 
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
* 
*             \96 floats     16 floats
* -------------------------------
* |         -----------------        |        8 floats
* |         |                 |        |
* |    apron|     constant     |        |        32 floats  => __shared__ float src_frag[32][80]
* |         |                 |        |
* |         -----------------        |
* -------------------------------
*/
void cu_sConv2_r816_exact(float4*                src, 
                          float4*                dst,
                          const uint            pitch_src, 
                          const uint            pitch_dst,
                          const uint            total_ker_len, 
                          const uint            Wker)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[32][96 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    glo_dex += 16 * pitch_src;
    reg_1 = src[glo_dex];

    store_to_shmem_L
    
    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        glo_dex += 16 * pitch_src;
        reg_1 = src[glo_dex];

        store_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)Const_Mem)[i];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}




__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 8 + 16 * 2)*(16 + 8 * 2) 即shmem half[32][160]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dimension of Dsrc is +16 * 2 = 32floats (+8 float4s if considered float4) on width
* The dimension of Dsrc is +8 * 2 = 16floats on height
* 
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
* 
*     \160 halfs (80floats)     16 halfs (20 float4s = (16 + 4) float4s)
* -------------------------------
* |         -----------------        |        8 halfs
* |         |                 |        |
* |    apron|     constant     |        |        32 floats  => __shared__ float src_frag[32][160]
* |         |                 |        |
* |         -----------------        |
* -------------------------------
*/
void cu_hConv2_r816_exact(float4*                src, 
                          float4*                dst,
                          const uint            pitch_src, 
                          const uint            pitch_dst,
                          const uint            total_ker_len, 
                          const uint            Wker)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[32][160 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    glo_dex += 16 * pitch_src;
    *((float4*)&reg_1) = src[glo_dex];

    hstore_to_shmem_L
    
    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        glo_dex += 16 * pitch_src;
        *((float4*)&reg_1) = src[glo_dex];

        hstore_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = ((__half*)Const_Mem)[i];
        tmp_ker.y = tmp_ker.x;

        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&reg_1);
#endif
}




__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem[32][80]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions(if float4 is consider horizentally, then +4 at width)
*
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度在8以内
* */
void cu_sConv2_r816_within(float4*                src, 
                           float4*                dst,
                           const uint            pitch_src, 
                           const uint            pitch_dst,
                           const uint            total_ker_len, 
                           const uint            Wker,
                           const int2            kernel_shift)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[32][96 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    glo_dex += 16 * pitch_src;
    reg_1 = src[glo_dex];

    store_to_shmem_L
    
    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        glo_dex += 16 * pitch_src;
        reg_1 = src[glo_dex];

        store_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + i % Wker;
        if (dy == kernel_shift.y) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)Const_Mem)[i];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}


__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem[32][80]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions(if float4 is consider horizentally, then +4 at width)
*
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度在8以内
* */
void cu_hConv2_r816_within(float4*                src, 
                           float4*                dst,
                           const uint            pitch_src, 
                           const uint            pitch_dst,
                           const uint            total_ker_len, 
                           const uint            Wker,
                           const int2            kernel_shift)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[32][160 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    glo_dex += 16 * pitch_src;
    *((float4*)&reg_1) = src[glo_dex];

    hstore_to_shmem_L
    
    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        glo_dex += 16 * pitch_src;
        *((float4*)&reg_1) = src[glo_dex];

        hstore_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = ((__half*)Const_Mem)[i];
        tmp_ker.y = tmp_ker.x;

        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&reg_1);
#endif
}


}



// ---------------------------------------- 16 x 8 -------------------------------------------------------------



extern "C"
{
__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem float[48][80]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 16 * 2 = 32 on all directions(if float4 is consider horizentally, then +8 at width)
* 
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
* 
*             \80 floats     8 floats
* -----------------------
* |           apron        |        16 floats
* |     ----------------    |
* |     |                |    |
* |     |     constant    |    |        48 floats  => __shared__ float src_frag[32][80]
* |     |                |    |
* |     ----------------    |
* |                        |
* -----------------------
*/
void cu_sConv2_r168_exact(float4*            src,
                         float4*            dst,
                         const uint            pitch_src,
                         const uint            pitch_dst,
                         const uint            total_ker_len,
                         const uint            Wker)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[48][80 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(32)

    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)Const_Mem)[i];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}




__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x8个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 8 + 8 * 2)*(16 + 8 * 2) 即shmem half[48][144]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 16 * 2 = 32 on all directions(if float4 is consider horizentally, then +8 at width)
* 
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
* 
*     \144 halfs (72 floats)    8 halfs (16 + 2)float4s
* -----------------------
* |           apron        |        16 halfs
* |     ----------------    |
* |     |                |    |
* |     |     constant    |    |        48 halfs  => __shared__ float src_frag[48][144]
* |     |                |    |
* |     ----------------    |
* |                        |
* -----------------------
*/
void cu_hConv2_r168_exact(float4*            src,
                         float4*            dst,
                         const uint            pitch_src,
                         const uint            pitch_dst,
                         const uint            total_ker_len,
                         const uint            Wker)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[48][144 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(32)

    if (threadIdx.y < 2) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = ((__half*)Const_Mem)[i];
        tmp_ker.y = tmp_ker.x;

        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&reg_1);
#endif
}





__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem[32][80]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions(if float4 is consider horizentally, then +4 at width)
*
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度在8以内
* */
void cu_sConv2_r168_within(float4*                src, 
                           float4*                dst,
                           const uint            pitch_src, 
                           const uint            pitch_dst,
                           const uint            total_ker_len, 
                           const uint            Wker,
                           const int2            kernel_shift)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[48][80 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(32)

    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + i % Wker;
        if (dy == kernel_shift.y) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)Const_Mem)[i];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}



__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem[32][80]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions(if float4 is consider horizentally, then +4 at width)
*
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度在8以内
* */
void cu_hConv2_r168_within(float4*                src,
                           float4*                dst,
                           const uint            pitch_src,
                           const uint            pitch_dst,
                           const uint            total_ker_len,
                           const uint            Wker,
                           const int2            kernel_shift)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[48][144 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(32)

    if (threadIdx.y < 2) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = ((__half*)Const_Mem)[i];
        tmp_ker.y = tmp_ker.x;

        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&reg_1);
#endif
}



__global__
void cu_sConv2_r8_exact_offset(float4*            src, 
                               float4*            dst,
                               const uint        pitch_src, 
                               const uint        pitch_dst,
                               const uint        total_ker_len, 
                               const uint        Wker,
                               const size_t        offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[32][80 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    glo_dex += 16 * pitch_src;
    reg_1 = src[glo_dex];

    store_to_shmem_L
    
    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        glo_dex += 16 * pitch_src;
        reg_1 = src[glo_dex];

        store_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];

        }
        tmp_ker = ((float*)&Const_Mem)[i + offset];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}





__global__
void cu_hConv2_r8_exact_offset(float4*            src, 
                               float4*            dst,
                               const uint        pitch_src, 
                               const uint        pitch_dst,
                               const uint        total_ker_len, 
                               const uint        Wker,
                               const size_t     offset)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[32][144 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    glo_dex += 16 * pitch_src;
    *((float4*)&reg_1) = src[glo_dex];

    hstore_to_shmem_L
    
    if (threadIdx.y < 2) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        glo_dex += 16 * pitch_src;
        *((float4*)&reg_1) = src[glo_dex];

        hstore_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = ((__half*)Const_Mem)[i + offset];
        tmp_ker.y = tmp_ker.x;
        
        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&reg_1);
#endif
}


__global__
void cu_sConv2_r8_within_offset(float4*                src, 
                                float4*                dst,
                                const uint            pitch_src, 
                                const uint            pitch_dst,
                                const uint            total_ker_len, 
                                const uint            Wker,
                                const int2            kernel_shift,
                                const size_t        offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[32][80 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    glo_dex += 16 * pitch_src;
    reg_1 = src[glo_dex];

    store_to_shmem_L
    
    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        glo_dex += 16 * pitch_src;
        reg_1 = src[glo_dex];

        store_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)Const_Mem)[i + offset];
        
        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }
    
    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;
}



__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x8个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 8 + 8 * 2)*(16 + 8 * 2) 即shmem half[32][144] -> float[32][72]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions
*
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
*
*         \144halfs(72 floats)     8 halfs
* -------------------------------
* |                                |        8 halfs
* |         -----------------        |
* |         |                 |        |
* |    apron|     constant     |        |        
* |         |                 |        |
* |         -----------------        |
* |                                |
* -------------------------------
*/
void cu_hConv2_r8_within_offset(float4*                src, 
                                float4*                dst,
                                const uint            pitch_src, 
                                const uint            pitch_dst,
                                const uint            total_ker_len, 
                                const uint            Wker,
                                const int2            kernel_shift,
                                const size_t        offset)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[32][144 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    glo_dex += 16 * pitch_src;
    *((float4*)&reg_1) = src[glo_dex];

    hstore_to_shmem_L
    
    if (threadIdx.y < 2) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        glo_dex += 16 * pitch_src;
        *((float4*)&reg_1) = src[glo_dex];

        hstore_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = ((__half*)Const_Mem)[i + offset];
        tmp_ker.y = tmp_ker.x;
        
        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    /*for (int i = 0; i < 8; ++i) {
        ((__half*)&reg_1)[i] = 255.f;
    }*/
    dst[glo_dex] = *((float4*)&reg_1);
#endif
}


}


// ---------------------------------------- 16 x 16 -------------------------------------------------------------




extern "C"
{
__global__
void cu_sConv2_r16_exact_offset(float4*                src, 
                                float4*                dst,
                                const uint            pitch_src, 
                                const uint            pitch_dst,
                                const uint            total_ker_len, 
                                const uint            Wker,
                                const size_t        offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[48][96 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(32)

    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)Const_Mem)[i + offset];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}



__global__
void cu_hConv2_r16_exact_offset(float4*                src, 
                                float4*                dst,
                                const uint            pitch_src, 
                                const uint            pitch_dst,
                                const uint            total_ker_len, 
                                const uint            Wker,
                                const size_t        offset)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[48][160 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(32)

    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = ((__half*)Const_Mem)[i + offset];
        tmp_ker.y = tmp_ker.x;

        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&reg_1);
#endif
}



__global__
void cu_sConv2_r16_within_offset(float4*        src, 
                                 float4*        dst,
                                 const uint        pitch_src, 
                                 const uint        pitch_dst,
                                 const uint        total_ker_len, 
                                 const uint        Wker,
                                 const int2        kernel_shift,
                                 const size_t    offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float src_frag[48][96 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(32)

    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(0)

            glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(16)

            glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)Const_Mem)[i + offset];
        
        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}


__global__
void cu_hConv2_r16_within_offset(float4*            src, 
                                 float4*            dst,
                                 const uint            pitch_src, 
                                 const uint            pitch_dst,
                                 const uint            total_ker_len, 
                                 const uint            Wker,
                                 const int2            kernel_shift,
                                 const size_t        offset)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[48][160 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(32)

    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = ((__half*)Const_Mem)[i + offset];
        tmp_ker.y = tmp_ker.x;

        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&reg_1);
#endif
}


}




// ---------------------------------------- 8 x 16 -------------------------------------------------------------


extern "C"
{
__global__
void cu_sConv2_r816_exact_offset(float4*            src, 
                                 float4*            dst,
                                 const uint            pitch_src, 
                                 const uint            pitch_dst,
                                 const uint            total_ker_len, 
                                 const uint            Wker,
                                 const size_t        offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[32][96 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    glo_dex += 16 * pitch_src;
    reg_1 = src[glo_dex];

    store_to_shmem_L
    
    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        glo_dex += 16 * pitch_src;
        reg_1 = src[glo_dex];

        store_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)Const_Mem)[i + offset];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}




__global__
void cu_hConv2_r816_exact_offset(float4*            src, 
                                 float4*            dst,
                                 const uint            pitch_src, 
                                 const uint            pitch_dst,
                                 const uint            total_ker_len, 
                                 const uint            Wker,
                                 const size_t        offset)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[32][160 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    glo_dex += 16 * pitch_src;
    *((float4*)&reg_1) = src[glo_dex];

    hstore_to_shmem_L
    
    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        glo_dex += 16 * pitch_src;
        *((float4*)&reg_1) = src[glo_dex];

        hstore_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = ((__half*)Const_Mem)[i + offset];
        tmp_ker.y = tmp_ker.x;

        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&reg_1);
#endif
}




__global__
void cu_sConv2_r816_within_offset(float4*                src, 
                                  float4*                dst,
                                  const uint            pitch_src, 
                                  const uint            pitch_dst,
                                  const uint            total_ker_len, 
                                  const uint            Wker,
                                  const int2            kernel_shift,
                                  const size_t            offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[32][96 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    glo_dex += 16 * pitch_src;
    reg_1 = src[glo_dex];

    store_to_shmem_L
    
    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        glo_dex += 16 * pitch_src;
        reg_1 = src[glo_dex];

        store_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + i % Wker;
        if (dy == kernel_shift.y) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)Const_Mem)[i + offset];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}


__global__
void cu_hConv2_r816_within_offset(float4*                src, 
                                  float4*                dst,
                                  const uint            pitch_src, 
                                  const uint            pitch_dst,
                                  const uint            total_ker_len, 
                                  const uint            Wker,
                                  const int2            kernel_shift,
                                  const size_t            offset)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[32][160 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    glo_dex += 16 * pitch_src;
    *((float4*)&reg_1) = src[glo_dex];

    hstore_to_shmem_L
    
    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        glo_dex += 16 * pitch_src;
        *((float4*)&reg_1) = src[glo_dex];

        hstore_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = ((__half*)Const_Mem)[i + offset];
        tmp_ker.y = tmp_ker.x;

        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&reg_1);
#endif
}


}



// ---------------------------------------- 16 x 8 -------------------------------------------------------------



extern "C"
{
__global__
void cu_sConv2_r168_exact_offset(float4*            src,
                                 float4*            dst,
                                 const uint            pitch_src,
                                 const uint            pitch_dst,
                                 const uint            total_ker_len,
                                 const uint            Wker,
                                 const size_t        offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[48][80 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(32)

    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)Const_Mem)[i + offset];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}




__global__
void cu_hConv2_r168_exact_offset(float4*            src,
                                 float4*            dst,
                                 const uint            pitch_src,
                                 const uint            pitch_dst,
                                 const uint            total_ker_len,
                                 const uint            Wker,
                                 const size_t        offset)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[48][144 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(32)

    if (threadIdx.y < 2) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = ((__half*)Const_Mem)[i + offset];
        tmp_ker.y = tmp_ker.x;

        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&reg_1);
#endif
}





__global__
void cu_sConv2_r168_within_offset(float4*                src, 
                                  float4*                dst,
                                  const uint            pitch_src, 
                                  const uint            pitch_dst,
                                  const uint            total_ker_len, 
                                  const uint            Wker,
                                  const int2            kernel_shift,
                                  const size_t            offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[48][80 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(32)

    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + i % Wker;
        if (dy == kernel_shift.y) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)Const_Mem)[i + offset];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}



__global__
void cu_hConv2_r168_within_offset(float4*                src,
                                  float4*                dst,
                                  const uint            pitch_src,
                                  const uint            pitch_dst,
                                  const uint            total_ker_len,
                                  const uint            Wker,
                                  const int2            kernel_shift,
                                  const size_t            offset)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[48][144 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(32)

    if (threadIdx.y < 2) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = ((__half*)Const_Mem)[i + offset];
        tmp_ker.y = tmp_ker.x;

        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&reg_1);
#endif
}


}


#endif