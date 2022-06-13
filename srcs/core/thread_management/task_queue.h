/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/

#ifndef _TASK_QUEUE_H_
#define _TASK_QUEUE_H_  

#include "../basic.h"
#include <queue>


namespace decx
{
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


#endif