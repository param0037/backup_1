/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _REARRANGEMENT_H_
#define _REARRANGEMENT_H_


#include "../../classes/Matrix.h"
#include "../../classes/GPU_Matrix.h"
#include "../../classes/MatrixArray.h"
#include "../../classes/GPU_MatrixArray.h"


namespace decx
{
    /**
    * @param __x : the destinated matrix
    * @param dst_dim .x -> width; .y -> height; .z -> MatNum
    */
    template <typename T>
    static void _dev_conv2_dst_rearrangement(_GPU_MatrixArray<T>* __x, uint3 dst_dim);


    /**
    * @param __x : the destinated matrix
    * @param dst_dim .x -> width; .y -> height
    */
    template <typename T>
    static void _dev_conv2_dst_rearrangement(_GPU_Matrix<T>* __x, uint2 dst_dim);


    /**
    * @param __x : the destinated matrix
    * @param dst_dim .x -> width; .y -> height; .z -> _store_type
    */
    template <typename T>
    static void conv2_dst_rearrangement(_Matrix<T>* __x, const uint3 dst_dim);


    /**
    * @param __x : the _MatrixArray wait to be rearranged
    * @param dst_dim .x -> width; .y -> height; .z -> ArrayNum; .w -> _store_type
    */
    template <typename T>
    static void conv2_mc_dst_rearrangement(_MatrixArray<T>* __x, uint4 dst_dim);
}



template <typename T>
static void decx::conv2_mc_dst_rearrangement(_MatrixArray<T>* __x, uint4 dst_dim)
{
    __x->re_construct(dst_dim.x, dst_dim.y, dst_dim.z, dst_dim.w);
}



template <typename T>
static void decx::_dev_conv2_dst_rearrangement(_GPU_MatrixArray<T>* __x, uint3 dst_dim)
{
    __x->re_construct(dst_dim.x, dst_dim.y, dst_dim.z);
}



template <typename T>
static void decx::conv2_dst_rearrangement(_Matrix<T>* __x, const uint3 dst_dim)
{
    __x->re_construct(dst_dim.x, dst_dim.y, dst_dim.z);
}


template <typename T>
static void decx::_dev_conv2_dst_rearrangement(_GPU_Matrix<T>* __x, uint2 dst_dim)
{
    __x->re_construct(dst_dim.x, dst_dim.y);
}


#endif        // #ifndef _REARRANGEMENT_H_