/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#include "complex_dev_funcs.cuh"


#define FFT_BLOCK_THREADS_LIMIT 512

#define Pi     3.1415926f
#define Two_Pi 6.2831853f


namespace decx
{
    namespace fft
    {
        /**
        * 拆解成质数核
        * __x apply the in-place operation e.g. /=, 循环完后判断__xs是否为1
        * 若为一，则说明可以分解为以2, 3, 5, 7的基，返回true，否则返回false，并储存__x
        */
        bool apart(int __x, std::vector<int>* res_arr);


        bool check_apart(int __x);


        struct FFT_Configs;


        struct FFT2D_kernel_launch_params;


        static void FFT_MixRadix_init(
            const size_t src_len, std::vector<int>* primes, decx::PtrInfo<int>* info, de::DH* handle);
    }
}




bool decx::fft::apart(int __x, std::vector<int>* res_arr)
{
    int prime[4] = { 5, 4, 3, 2 };
    int tmp = 0;
    // 若判断完一轮全都找不到合适的，break掉while循环，出结果了
    bool __continue = true;
    bool round_not_f = true;

    while (__continue)
    {
        round_not_f = true;
        for (int i = 0; i < 4; ++i){
            if ((__x % prime[i]) == 0) {
                (*res_arr).push_back(prime[i]);
                round_not_f = false;
                __x /= prime[i];
                break;
            }
        }
        if (round_not_f) {    // 如果一轮中没有找到合适的
            __continue = false;
        }
    }
    if (__x != 1) {      // 说明__x无法完全分解
        (*res_arr).push_back(__x);
        return false;
    }
    else {
        return true;
    }
}



bool decx::fft::check_apart(int __x)
{
    int prime[4] = { 5, 4, 3, 2 };
    int tmp = 0;
    // 若判断完一轮全都找不到合适的，break掉while循环，出结果了
    bool __continue = true;
    bool round_not_f = true;

    while (__continue)
    {
        round_not_f = true;
        for (int i = 0; i < 4; ++i) {
            if ((__x % prime[i]) == 0) {
                round_not_f = false;
                __x /= prime[i];
                break;
            }
        }
        if (round_not_f) {    // 如果一轮中没有找到合适的
            __continue = false;
        }
    }
    if (__x != 1) {      // 说明__x无法完全分解
        return false;
    }
    else {
        return true;
    }
}




struct decx::fft::FFT_Configs
{
    std::vector<int> _base;
    decx::PtrInfo<int> index_interpret_table;
    int _base_num;
    
    int _consecutive_4;

    /*
    * If it is configuring FFT2D, and this pointer will be activated. Otherwise,
    * it remains nullptr. Call this->FFT2D_launch_param_gen() to configure these params
    */
    decx::fft::FFT2D_kernel_launch_params* _LP_FFT2D;

    bool _is_2s;        // If the length is 2's power
    bool _can_halve;    // If it can be halved, so the kernel can load datas in float4 with Load.128

    FFT_Configs() : _LP_FFT2D(nullptr)
    {}

    void FFT1D_config_gen(const size_t vec_len, de::DH* handle, cudaStream_t& S);

    /*
    * Call only when it is used to configure a 2D (I)FFT process
    */
    void FFT2D_launch_param_gen(const int vec_len, const int _height);


    void release_inner_tmp();


    ~FFT_Configs();
};



decx::fft::FFT_Configs::~FFT_Configs()
{
    this->release_inner_tmp();
    delete this->_LP_FFT2D;
}



struct decx::fft::FFT2D_kernel_launch_params
{
    int _Wsrc, _Hsrc;
    // ~[0] -> grid_params; ~[1] : block_params
    dim3 Sort[2],
         Radix_2[2],
         Radix_3[2],
         Radix_4[2],
         Radix_4_consec[2],
         Radix_5[2],
         Radix_2_last[2];
};



#define kernel_param_gen(_param_name, _clamp_crit) \
{    \
current_grid = &(this->_LP_FFT2D->_param_name[0]);   \
current_block = &(this->_LP_FFT2D->_param_name[1]);  \
current_block->y =   \
req_threads_num > _clamp_crit ?    \
_clamp_crit :   \
    decx::utils::ceil<int>(req_threads_num, 32) * 32;  \
current_block->x = current_block->y == _clamp_crit ? 1 : \
(_clamp_crit / current_block->y);    \
\
current_grid->x = decx::utils::ceil<int>(_height, current_block->x); \
current_grid->y = decx::utils::ceil<int>(req_threads_num, current_block->y);    \
}



void decx::fft::FFT_Configs::FFT2D_launch_param_gen(const int vec_len, const int _height)
{
    this->_LP_FFT2D = new decx::fft::FFT2D_kernel_launch_params();
    if (this->_LP_FFT2D == nullptr) {
        SetConsoleColor(4);
        printf("FFT_Configs new failed!\n");
        ResetConsoleColor;
        return;
    }
    this->_LP_FFT2D->_Wsrc = vec_len;
    this->_LP_FFT2D->_Hsrc = _height;

    dim3* current_grid, *current_block;

    int fragment_len = vec_len / 2;
    int req_threads_num = fragment_len;

    if (this->_can_halve) {
        kernel_param_gen(Radix_2_last, FFT_BLOCK_THREADS_LIMIT);
    }

    kernel_param_gen(Sort, FFT_BLOCK_THREADS_LIMIT);

    req_threads_num = fragment_len / 2;
    if (req_threads_num != 0) {
        kernel_param_gen(Radix_2, FFT_BLOCK_THREADS_LIMIT);
    }

    req_threads_num = fragment_len / 3;
    if (req_threads_num != 0){
        kernel_param_gen(Radix_3, FFT_BLOCK_THREADS_LIMIT);
    }

    req_threads_num = fragment_len / 4;
    if (req_threads_num != 0) {
        kernel_param_gen(Radix_4, FFT_BLOCK_THREADS_LIMIT);
    }

    req_threads_num = fragment_len / 5;
    if (req_threads_num != 0) {
        kernel_param_gen(Radix_5, FFT_BLOCK_THREADS_LIMIT);
    }
    if (this->_is_2s) {
        req_threads_num = fragment_len / 4;
        if (req_threads_num != 0)
        {
            //kernel_param_gen(Radix_4_consec, 256);
            current_grid = &(this->_LP_FFT2D->Radix_4_consec[0]);
            current_block = &(this->_LP_FFT2D->Radix_4_consec[1]);
            // 32x number of threads must be satisfied _CUDA_WARP_SIZE_
            current_block->y = 
                req_threads_num > 256 ? 256 : decx::utils::ceil<int>(req_threads_num, _CUDA_WARP_SIZE_) * _CUDA_WARP_SIZE_;
            current_block->x = current_block->y == 256 ? 1 : (256 / current_block->y);    
            
            current_grid->x = decx::utils::ceil<int>(_height, current_block->x);
            // 1 << (2 * this->_consecutive_4) is how many elements a block actually process
            current_grid->y = decx::utils::ceil<int>(fragment_len, (1 << (2 * this->_consecutive_4)));
        }
    }
}




void decx::fft::FFT_Configs::FFT1D_config_gen(const size_t vec_len, de::DH* handle, cudaStream_t &S)
{
    const int bin_len = _GetHighest(vec_len - 1);
    this->_is_2s = !(vec_len & (~(1 << bin_len)));

    // initialize the base_vector
    this->_base.clear();

    // If the vector_lenght is in 2's power, it will activate radix-2 and attempt to use radix-4 to
    // accelerate the calculation
    if (this->_is_2s) {
        this->_can_halve = true;
        // All the elements in the base will be either 2 or 4
        decx::fft::apart(vec_len / 2, &this->_base);
        this->_base_num = this->_base.size();
        decx::fft::FFT_MixRadix_init(vec_len / 2, &this->_base, &this->index_interpret_table, handle);

        int consec_4 = 0;
        if (this->_base[0] == 4) {
            int test = 4;
            consec_4 = 1;
            for (int i = 1; i < this->_base_num; ++i) {
                int tmp = this->_base[i];
                test *= tmp;
                if (test <= 1024 && tmp == 4) {
                    ++consec_4;
                }
                else { break; }
            }
        }
        this->_consecutive_4 = consec_4;
    }
    else {
        this->_can_halve = (vec_len % 2 == 0);

        if (this->_can_halve) {
            decx::fft::apart(vec_len / 2, &this->_base);
            this->_base_num = this->_base.size();
            decx::fft::FFT_MixRadix_init(vec_len / 2, &this->_base, &this->index_interpret_table, handle);
        }
        else {
            decx::fft::apart(vec_len, &this->_base);
            this->_base_num = this->_base.size();
            decx::fft::FFT_MixRadix_init(vec_len, &this->_base, &this->index_interpret_table, handle);
        }
    }
    checkCudaErrors(cudaMemcpyToSymbolAsync(Const_Mem, this->index_interpret_table.ptr, 
        this->_base_num * 3 * sizeof(int), 0, cudaMemcpyHostToDevice, S));

    decx::Success(handle);
}




void decx::fft::FFT_Configs::release_inner_tmp()
{
    if (!this->_is_2s) {
        decx::alloc::_dealloc_Hv(this->index_interpret_table.block);
    }
}




static
void decx::fft::FFT_MixRadix_init(
    const size_t src_len, std::vector<int>* primes, decx::PtrInfo<int>* info, de::DH* handle)
{
    int ori_base_len = primes->size();

    if (decx::alloc::_host_virtual_page_malloc<int>(info, ori_base_len * 3 * sizeof(int))) {
        decx::err::AllocateFailure(handle);
        return;
    }

    for (int i = 0; i < ori_base_len; ++i) {
        info->ptr[i] = primes->operator[](i);
    }

    // 正序基
    int* base_ptr = info->ptr + ori_base_len;
    int tmp = 1;
    for (int i = ori_base_len - 1; i > 0; --i) {
        tmp *= (*primes)[i];
        base_ptr[i - 1] = tmp;
    }
    base_ptr[ori_base_len - 1] = 1;

    base_ptr += ori_base_len;
    // 译序的结果和基的元素顺序有关，做FFT的时候不能乱
    // 反序基
    tmp = 1;
    for (int i = 0; i < ori_base_len - 1; ++i) {
        tmp *= (*primes)[i];
        base_ptr[ori_base_len - i - 2] = tmp;
    }
    base_ptr[ori_base_len - 1] = 1;
}




#define FFT_1D_SORT 512
 



__host__ __device__
void GetRev(int         __x,
            const int   bit_len,
            int*        bases,
            int*        rev_bases,
            int*        res_dex)
{
    *res_dex = 0;
    for (int i = 0; i < bit_len; ++i) {
        *res_dex += rev_bases[bit_len - 1 - i] * (__x / bases[i]);
        __x = __x % bases[i];
    }
}