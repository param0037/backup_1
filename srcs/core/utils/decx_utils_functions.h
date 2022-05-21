/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once


namespace decx
{
    namespace utils
    {
        /*
        * @brief : The '_abd' suffix means that the return value WILL NOT
        * plus one when the input value reaches the critical points (e.g.
        * 2, 4, 8, 16, 32...)
        */
        constexpr static int _GetHighest_abd(size_t __x) noexcept;

        /*
        * @brief : The '_abd' suffix means that the return value WILL
        * plus one when the input value reaches the critical points (e.g.
        * 2, 4, 8, 16, 32...)
        */
        constexpr static int _GetHighest(size_t __x) noexcept;


        /*
        * @return return __x < _boundary ? _boundary : __x;
        */
        template <typename _Ty>
        constexpr static size_t clamp_min(_Ty __x, _Ty _bpunary) noexcept;
        

        /*
        * @return return __x > _boundary ? _boundary : __x;
        */
        template <typename _Ty>
        constexpr static size_t clamp_max(_Ty __x, _Ty _bpunary) noexcept;


        /*
        * @return return (__deno % __numer) != 0 ? __deno / __numer + 1 : __deno / __numer;
        */
        template <typename _Ty>
        constexpr
        inline static _Ty ceil(_Ty __deno, _Ty __numer) noexcept;


#ifdef _DECX_CUDA_CODES_
        template <typename _Ty>
        __device__
        _Ty cu_ceil(_Ty __deno, _Ty __numer) {
            return (__deno / __numer) + (int)((bool)(__deno % __numer));
        }
#endif


        constexpr inline static int Iabs(int n) noexcept {
            return (n ^ (n >> 31)) - (n >> 31);
        }
    }
}



constexpr
static int decx::utils::_GetHighest_abd(size_t __x) noexcept
{
    --__x;
    int res = 0;
    while (__x) {
        ++res;
        __x >>= 1;
    }
    return res;
}


/*
* @return return __x < _boundary ? _boundary : __x;
*/
template <typename _Ty>
constexpr static size_t decx::utils::clamp_min(_Ty __x, _Ty _boundary) noexcept{
    return __x < _boundary ? _boundary : __x;
}

/*
* @return return __x > _boundary ? _boundary : __x;
*/
template <typename _Ty>
constexpr static size_t decx::utils::clamp_max(_Ty __x, _Ty _boundary) noexcept {
    return __x > _boundary ? _boundary : __x;
}


template <typename _Ty>
constexpr
inline static _Ty decx::utils::ceil(_Ty __deno, _Ty __numer) noexcept
{
    //return (__deno % __numer) != 0 ? __deno / __numer + 1 : __deno / __numer;
    return (__deno / __numer) + (int)((bool)(__deno % __numer));
}




static int _GetHighest(size_t __x) noexcept
{
    int res = 0;
    while (__x) {
        ++res;
        __x >>= 1;
    }
    return res;
}



constexpr
static int decx::utils::_GetHighest(size_t __x) noexcept
{
    int res = 0;
    while (__x) {
        ++res;
        __x >>= 1;
    }
    return res;
}




inline
static int _cu_ceil(int __deno, int __numer) noexcept
{
    return (__deno % __numer) != 0 ? __deno / __numer + 1 : __deno / __numer;
}



inline
static int _cu_ceil_size_t(size_t __deno, size_t __numer) noexcept
{
    return (__deno % __numer) != 0 ? __deno / __numer + 1 : __deno / __numer;
}