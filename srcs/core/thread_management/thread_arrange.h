/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/

#ifndef _THREAD_ARRANGE_H_
#define _THREAD_ARRANGE_H_

#include "../basic.h"

namespace decx
{
    namespace utils
    {
        typedef struct _thread_arrange_1D
        {
            bool is_avg;
            size_t _prev_proc_len;
            size_t _leftover;
            size_t _prev_len;

            _thread_arrange_1D(const uint thr_num, const size_t total_proc_len)
            {
                this->_prev_proc_len = total_proc_len / thr_num;

                if (total_proc_len % thr_num) {
                    this->is_avg = false;
                    this->_prev_len = this->_prev_proc_len * (thr_num - 1);
                    this->_leftover = total_proc_len - this->_prev_len;
                }
                else {
                    this->is_avg = true;
                    this->_prev_len = 0;
                    this->_leftover = 0;
                }
            }
        }_thr_1D;
    }
}



#endif