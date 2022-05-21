/**
*	---------------------------------------------------------------------
*	Author : Wayne
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/

#pragma once

#ifndef _THREAD_POOL_H_
#define _THREAD_POOL_H_


#include "../basic.h"

#define MAX_THREAD_NUM 16


// 用一个mutex管理一个task_queue，用另一个mutex来wait()
namespace decx
{
    class ThreadPool;

    class ThreadTaskQueue;
}


typedef std::packaged_task<void()> Task;


class decx::ThreadTaskQueue
{
private:
    void move_ahead();

public:
    // private variables for each thread
    std::mutex _mtx;
    std::condition_variable _cv;
    Task* _task_queue,
        * begin_ptr,
        * end_ptr;

    int _task_num;
    bool _shutdown;

    ThreadTaskQueue();

    template <class FuncType, class ...Args>
    std::future<void> emplace_back(FuncType&& f, Args&& ...args);

    void pop_back();

    void pop_front();


    ~ThreadTaskQueue();
};



void decx::ThreadTaskQueue::move_ahead()
{
    int idle_size = (int)(this->begin_ptr - this->_task_queue);

    memcpy(this->_task_queue, this->begin_ptr, idle_size * sizeof(Task));
    if (this->_task_num - idle_size > 0)
        memcpy(this->_task_queue + idle_size, this->begin_ptr + idle_size, (this->_task_num - idle_size) * sizeof(Task));
}



decx::ThreadTaskQueue::ThreadTaskQueue() {
    this->_task_queue = (Task*)malloc(64 * sizeof(Task));
    this->begin_ptr = this->_task_queue;
    this->end_ptr = this->_task_queue + 64;

    this->_shutdown = false;
    this->_task_num = 0;
}



template <class FuncType, class ...Args>
std::future<void> decx::ThreadTaskQueue::emplace_back(FuncType&& f, Args&& ...args) {
    new (this->begin_ptr + this->_task_num)Task(
        std::bind(std::forward<FuncType>(f), std::forward<Args>(args)...));

    ++this->_task_num;
    return (this->begin_ptr + this->_task_num - 1)->get_future();
}



void decx::ThreadTaskQueue::pop_back() {
    --this->_task_num;
}



void decx::ThreadTaskQueue::pop_front()
{
    if (this->_task_num > 1) {
        if (this->begin_ptr + 1 == this->end_ptr) {
            this->move_ahead();
            this->begin_ptr = this->_task_queue;
        }
        else {
            ++this->begin_ptr;
        }
    }
    else {
        this->begin_ptr = this->_task_queue;
    }
    --this->_task_num;
}



decx::ThreadTaskQueue::~ThreadTaskQueue() {
    free(this->_task_queue);
}



class decx::ThreadPool
{
private:
    std::thread* _thr_list;

    decx::ThreadTaskQueue* _task_schd;
    std::mutex _mtx;

    size_t _max_thr_num, current_thread_num;
    bool _all_shutdown;

    void _find_task_queue_id(size_t* id);

    // main_loop callback function running on each thread
    void _thread_main_loop(const size_t pool_id);

public:
    size_t _hardware_concurrent;

    void Start();

    ThreadPool(const int thread_num, const bool start_at_begin);

    template <class FuncType, class ...Args>
    std::future<void> register_task(FuncType&& f, Args&& ...args);


    template <class FuncType, class ...Args>
    std::future<void> register_task_by_id(size_t id, FuncType&& f, Args&& ...args);


    void add_thread(const int add_thread_num);


    void TerminateAllThreads();


    ~ThreadPool();
};




void decx::ThreadPool::_find_task_queue_id(size_t* id)
{
    size_t task_que_len = this->current_thread_num;
    size_t res_id = 0,
        least_len = (this->_task_schd)->_task_num;

    if (least_len != 0) {
        for (size_t i = 1; i < task_que_len; ++i)
        {
            decx::ThreadTaskQueue* tmp_iter = this->_task_schd + i;

            size_t current_len = tmp_iter->_task_num;

            if (current_len != 0) {
                if (current_len < least_len)
                    least_len = current_len;
            }
            else {
                least_len = i;
                break;
            }
        }
        *id = least_len;
    }
    else {
        *id = res_id;
    }
}



void decx::ThreadPool::_thread_main_loop(const size_t pool_id)
{
    decx::ThreadTaskQueue* thread_unit = &(this->_task_schd[pool_id]);
    while (!thread_unit->_shutdown) {
        {
            std::unique_lock<std::mutex> lock{ thread_unit->_mtx };

            while ((thread_unit->_task_num == 0) && (!thread_unit->_shutdown)) {
                thread_unit->_cv.wait(lock);
            }
        }

        if (thread_unit->_task_num != 0) {
            Task* task = thread_unit->begin_ptr + thread_unit->_task_num - 1;
            (*task)();     // execute the tast

            thread_unit->_mtx.lock();
            thread_unit->pop_back();
            thread_unit->_mtx.unlock();
        }
    }
    return;
}



void decx::ThreadPool::Start()
{
    // 仅创建可以 concurrent 的线程
    this->_all_shutdown = false;

    for (int i = 0; i < this->current_thread_num; ++i) {
        new(this->_task_schd + i) decx::ThreadTaskQueue();
    }
    for (size_t i = 0; i < this->current_thread_num; ++i) {
        new(this->_thr_list + i) std::thread(&decx::ThreadPool::_thread_main_loop, this, i);
    }
}



decx::ThreadPool::ThreadPool(const int thread_num, const bool start_at_begin)
{
    this->_all_shutdown = true;
    this->_max_thr_num = MAX_THREAD_NUM;
    this->current_thread_num = thread_num;

    this->_hardware_concurrent = std::thread::hardware_concurrency();

    this->_task_schd = (decx::ThreadTaskQueue*)malloc(this->_max_thr_num * sizeof(decx::ThreadTaskQueue));
    this->_thr_list = (std::thread*)malloc(this->_max_thr_num * sizeof(std::thread));

    if (start_at_begin) {
        Start();
    }
}



template <class FuncType, class ...Args>
std::future<void> decx::ThreadPool::register_task(FuncType&& f, Args&& ...args)
{
    size_t id;
    this->_find_task_queue_id(&id);

    decx::ThreadTaskQueue* tmp_task_que = &(this->_task_schd[id]);
    tmp_task_que->_mtx.lock();
    std::future<void> fut = tmp_task_que->emplace_back(std::forward<FuncType>(f), std::forward<Args>(args)...);
    tmp_task_que->_mtx.unlock();
    tmp_task_que->_cv.notify_one();

    return fut;
}




template <class FuncType, class ...Args>
std::future<void> decx::ThreadPool::register_task_by_id(size_t id, FuncType&& f, Args&& ...args)
{
    decx::ThreadTaskQueue* tmp_task_que = &(this->_task_schd[id]);
    tmp_task_que->_mtx.lock();
    std::future<void> fut = tmp_task_que->emplace_back(std::forward<FuncType>(f), std::forward<Args>(args)...);
    tmp_task_que->_mtx.unlock();
    tmp_task_que->_cv.notify_one();

    return fut;
}




void decx::ThreadPool::add_thread(const int add_thread_num)
{
    if (this->current_thread_num + add_thread_num > this->_max_thr_num) {
        return;
    }
    else {
        for (int i = 0; i < add_thread_num; ++i) {
            new(this->_task_schd + this->current_thread_num + i) decx::ThreadTaskQueue();
        }
        for (size_t i = 0; i < add_thread_num; ++i) {
            new(this->_thr_list + this->current_thread_num + i) std::thread(
                &decx::ThreadPool::_thread_main_loop, this, i);
        }
    }
}




void decx::ThreadPool::TerminateAllThreads()
{
    for (int i = 0; i < this->current_thread_num; ++i) {
        std::thread* _iter = this->_thr_list + i;
        decx::ThreadTaskQueue* Tschd_iter = this->_task_schd + i;
        {
            std::unique_lock<std::mutex> lck(Tschd_iter->_mtx);
            Tschd_iter->_shutdown = true;
        }
        Tschd_iter->_cv.notify_one();
        _iter->join();
    }

    this->_all_shutdown = true;
}



decx::ThreadPool::~ThreadPool() {
    if (!this->_all_shutdown) {
        TerminateAllThreads();
    }
    for (int i = 0; i < this->current_thread_num; ++i) {
        std::thread* _iter = this->_thr_list + i;
        decx::ThreadTaskQueue* Tschd_iter = this->_task_schd + i;
        Tschd_iter->~ThreadTaskQueue();
        _iter->~thread();
    }

    free(this->_task_schd);
    free(this->_thr_list);
}



namespace decx
{
    extern decx::ThreadPool thread_pool(std::thread::hardware_concurrency(), true);
}


#define _THREAD_FUNCTION_


#endif