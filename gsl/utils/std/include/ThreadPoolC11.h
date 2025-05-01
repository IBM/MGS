// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

/*
MIT License

Copyright (c) 2016 Mariano Trebino

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */
#ifndef THREAD_POOL_C11_H
#define THREAD_POOL_C11_H
//#pragma once

#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

// Thread safe implementation of a Queue using a std::queue
template <typename T>
class SafeQueue 
{
private:
  std::queue<T> m_queue;
  std::mutex m_mutex;
public:
  SafeQueue() 
  {

  }

  SafeQueue(SafeQueue& other) 
  {
    //TODO:
  }

  ~SafeQueue() 
  {

  }


  bool empty() 
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    return m_queue.empty();
  }
  
  int size() 
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    return m_queue.size();
  }

  void enqueue(T& t) 
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_queue.push(t);
  }
  
  bool dequeue(T& t) 
  {
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_queue.empty()) 
    {
      return false;
    }
    t = std::move(m_queue.front());
    
    m_queue.pop();
    return true;
  }
};

class ThreadPoolC11 {
private:
  class ThreadWorker {
  private:
    int m_id;
    ThreadPoolC11 * m_pool;
  public:
    ThreadWorker(ThreadPoolC11 * pool, const int id)
      : m_pool(pool), m_id(id) {
    }

    void operator()() {
      std::function<void()> func;
      bool dequeued;
      while (!m_pool->m_shutdown) {
        {
          std::unique_lock<std::mutex> lock(m_pool->m_conditional_mutex);
          /* put the current thread to block until 
              the condition variable 'm_cv_task' is 
              notified by (1)  m_cv_task.notifiy_[one, all]called by another thread
                          (2) spurious wakeup occurs
            if it blocks the current thread, it first release the mutex (for other threads to use)
            and add it to the list of threads waiting on *this (i.e. the condition variable)

            once the thread is unblocked, regardless of the reason (1) or (2)
              the lock on the mutx is also re-acquired [until C++14]
           
           if unblocked, check for Predicate to decide whether keep blocking or not
           if (! Pred)
           {
            wait(lock);
           }
           */

          /* deadlock because it can miss the signal --> wait forever */
          //m_pool->m_cv_task.wait(lock); 

          /* deadlock because there is no job but we cannot shutdown
          while (!pred()) { //continue to wait if pred() returns false
              wait(lock);
          }
          while (queue.empty) --> while (!  (! queue.empty))
          {
            wait()
          } --> continue to wait() until queue is NOT empty
           */
          //m_pool->m_cv_task.wait(lock, [this](){ return (! m_pool->m_queue.empty()); });

          /* deadlock because there is no job but we cannot shutdown
          while ( queue.empty() || not shutdown )
          {
            wait()
          }
           */
          //m_pool->m_cv_task.wait(lock, [this](){ return (! (m_pool->m_queue.empty() && ! m_pool->m_shutdown)); });
          /* NO deadlock 
          while (queue.empty and ! shutdown)  --> while(! )
          {
            wait()
          }

           */
          //m_pool->m_cv_task.wait(lock, [this](){ return (! (! m_pool->m_shutdown && m_pool->m_queue.empty())); });
          m_pool->m_cv_task.wait(lock, [this](){ return (! (m_pool->m_queue.empty() && ! (m_pool->m_shutdown) )); });
          //if (m_pool->m_queue.empty()) {
          //  /* this thread is blocked until 'm_cv_task' is notified */
          //  m_pool->m_cv_task.wait(lock);
          //}
          dequeued = m_pool->m_queue.dequeue(func);
          if (dequeued) {
            m_pool->busy++;
          }
        }
        if (dequeued) {
          // release lock. run async
          //lock.unlock();
          func();
          std::lock_guard<std::mutex> lock(m_pool->m_conditional_mutex);
          //lock.lock();
          m_pool->busy--;
          m_pool->m_cv_finished.notify_one();
        }
      }
    }
  };

  bool m_shutdown;
  bool m_isInit;
  size_t _nThreads;
  unsigned int busy;
  std::vector<std::thread> m_threads;
  SafeQueue<std::function<void()>> m_queue;
  std::mutex m_conditional_mutex;
  //std::mutex m_mutex_finished;
  /* condition variable on telling for if a new task comes */
  std::condition_variable m_cv_task;
  /* condition variable on telling if all task completed */
  std::condition_variable m_cv_finished;
public:
  ThreadPoolC11(const int n_threads)
    : m_shutdown(false), m_isInit(false),
    _nThreads(n_threads), busy(0), m_threads(std::vector<std::thread>(n_threads)) 
  {
  }
  size_t size() { return _nThreads; }

  ThreadPoolC11(const ThreadPoolC11 &) = delete;
  ThreadPoolC11(ThreadPoolC11 &&) = delete;

  ThreadPoolC11 & operator=(const ThreadPoolC11 &) = delete;
  ThreadPoolC11 & operator=(ThreadPoolC11 &&) = delete;

  // Inits thread pool
  //   .. before any 'submit' 
  //   .. and after 'shutdown()' is used
  void init() {
    assert(m_isInit == false);
    m_shutdown = false;
    for (int i = 0; i < m_threads.size(); ++i) {
      m_threads[i] = std::thread(ThreadWorker(this, i));
    }
    m_isInit = true;
  }

  // Waits until threads finish their current task and shutdowns the pool
  void shutdown() {
    {
      /* to prevent missing the signal, lock the mutex before notifying to condition variable */
      std::lock_guard<std::mutex> lock(m_conditional_mutex);
      m_shutdown = true;
      m_cv_task.notify_all();
    }
    
    for (int i = 0; i < m_threads.size(); ++i) {
      if(m_threads[i].joinable()) {
        m_threads[i].join();
      }
    }
    m_isInit = false;
  }
  //void waitFinished()
  //{
  //  /*
  //  while(!m_queue.empty()) 
  //    //check if there are any tasks in queue waiting to be picked up
  //    {
  //    //do literally nothing
  //  }
  //  */
  //  std::unique_lock<std::mutex> lock(m_conditional_mutex);
  //  //std::unique_lock<std::mutex> lock(m_mutex_finished);
  //  m_cv_finished.wait(lock, [this](){ return m_queue.empty() && (busy == 0); });
  //  //std::lock_guard<std::mutex> lock(m_conditional_mutex);
  //}
  bool finishedJobs()
  {
    std::lock_guard<std::mutex> lock(m_conditional_mutex);
    //if (m_queue.empty())
    //  std::cout << "busy " << busy << std::endl;
    //std::cout << "  queue size " << m_queue.size() << ", result: " << m_queue.empty() << "   " << busy << std::endl;
    return m_queue.empty() && (busy == 0); 
  }

  // Submit a function to be executed asynchronously by the pool
  template<typename F, typename...Args>
  auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
    assert(m_isInit);
    // Create a function with bounded parameters ready to execute
    std::function<decltype(f(args...))()> func = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    // Encapsulate it into a shared ptr in order to be able to copy construct / assign 
    auto task_ptr = std::make_shared<std::packaged_task<decltype(f(args...))()>>(func);

    // Wrap packaged task into void function
    std::function<void()> wrapper_func = [task_ptr]() {
      (*task_ptr)(); 
    };

    // Enqueue generic wrapper function
    m_queue.enqueue(wrapper_func);

    // Wake up one thread if its waiting
    m_cv_task.notify_one();

    // Return future from promise
    return task_ptr->get_future();
  }
};


#endif
