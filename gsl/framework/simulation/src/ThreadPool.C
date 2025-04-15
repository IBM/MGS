// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ThreadPool.h"
#ifndef DISABLE_PTHREADS

#include "WorkUnit.h"

#include <errno.h>
#include <unistd.h>
#include <iostream>

#ifdef AIX
#ifdef _PRINT
#define PRINT(a,b) printState(a,b);
#else
#define PRINT(a,b)
#endif
#ifdef _EXECUTE_TIMINGS_PRINT
#define EXECUTE_TIMINGS_PRINT(a,b) printState(a,b);
#else
#define EXECUTE_TIMINGS_PRINT(a,b)
#endif

static void exit_on_error(int status,const char *str);
static int report_on_error(int status,const char *str);

int report_on_error(int status,const char *str)
{
   if(status != 0) {
      std::cerr <<"Error, status = "<<status<< str <<" "<<std::endl;
      switch (status) {
         case EINVAL:
            std::cerr << "EINVAL" << std::endl;
            break;
         case ESRCH:
            std::cerr << "ESRCH" << std::endl;
            break;
         case EDEADLK:
            std::cerr << "EDEADLK" << std::endl;
            break;
         case EAGAIN:
            std::cerr << "EAGAIN" << std::endl;
            break;
         case ENOMEM:
            std::cerr << "ENOMEM" << std::endl;
            break;
         case EPERM:
            std::cerr << "EPERM" << std::endl;
            break;
      }
   }
   return status;
}

void exit_on_error(int status,const char *str)
{
   if (report_on_error(status, str)) exit(-1);
}

ThreadPool::ThreadPool(int N, int numCpus, bool bindThreadsToCpus)
   : _numThreads(N), _numCpus(numCpus)
{
   // Constructor argument N determines the number of threads this pool will
   // create. If N is zero, the thread pool will create
   // _numCpus threads, where _numCpus is the number of cpus online.

   // Initialize the mutexes, cond wait variables etc.
   // Do this before threads are created
   pthread_mutex_init(&_queueLock, NULL);
   pthread_cond_init(&_workDone, NULL);
   pthread_cond_init(&_workStart, NULL);
   pthread_cond_init(&_teamAssembled, NULL);
   pthread_mutex_init(&_coutLock, NULL);

   // Start with no work to do
   _workCount = 0;
   _threadsReady=_threadsActive=0;
   _workStartSent=0;
   _nextWorkIndex = 0;
   _threadsoutstanding = 0;
   next_thread_data_index = 0;

   // If the constructor argument, N is zero, set _numThreads to the number of CPUS
   if (N == 0) {
      _numThreads = _numCpus;
   }
   _timer.start();

   thread_data_array = new thread_data[_numThreads];
   for(int i = 0; i < _numThreads; i++) {
      thread_data_array[i].pthread_id = 0;
      #ifndef LINUX
      thread_data_array[i].kernel_thread_id = 0;
      #endif
      // Default is "to cancel" the thread
      thread_data_array[i].to_cancel = 1;
   }
   thread_ids = new pthread_t[_numThreads];

   // Use a barrier with 1 arrival, for thread creation
   barrier1 = new Barrier(1);

   // Keep creating worker threads until you get the desired num of unique kernel threads
   while(next_thread_data_index < _numThreads) {
      createThread();
      barrier1->wait();

      // Let the thread  determine its thread ids and put it into the array "thread_data_array"
      barrier1->go();
      barrier1->wait();

      // Find out if this thread has a unique kernel thread id
      // this results in setting the "to_cancel" item in the thread_data array
      findUniqueKernels(next_thread_data_index);
      // Let the thread go and cancel itself
      if(thread_data_array[next_thread_data_index].to_cancel) {
	 barrier1->go();
      } else {
         barrier1->go();
         barrier1->wait();
         // Wait until the thread increments the nex_thread_data_index (otherwise 
	 // we have a race condition -- the constructor thread may read the value 
	 // of next_thread_data_index at the same time the worker thread is 
	 // incrementing it.
         barrier1->go();
      }

   }
   #ifndef LINUX
   if (bindThreadsToCpus) {
      bindCpus(_numCpus);
   }
   #endif
}

// Creates a single pthread. This is put in the location thread_ids[next_thread_data_index]

void ThreadPool::createThread()
{
   int status = pthread_create(thread_ids+next_thread_data_index, NULL, CWorkerFriend, this);
   exit_on_error(status, "Thread creation error");
}

void ThreadPool::findUniqueKernels(int index)
{
   // We know that the unique kernels are in the first "index - 1" elements of the 
   // thread_data_array. We now compare the kernel id in 'index' to the ones before it.
   int is_unique = 1;
   #ifndef LINUX
   int i;

   for(i = 0; i <= index - 1; i++) {
      if(thread_data_array[i].kernel_thread_id == thread_data_array[index].kernel_thread_id) {
         is_unique = 0;
         break;
      }
   }
   #endif
   if(!is_unique) {
      thread_data_array[index].to_cancel = 1;
      // Unique kernel id.
   } else {
      thread_data_array[index].to_cancel = 0;
   }
   // Now let the worker thread routine take care of exiting itself if necessary. 
   // No need to do anything else here.
}

// Worker function for a thread
void *ThreadPool::worker()
{
   int tasknum;
   int error, status;
   int myWorkIndex = 0;          // work queue element processed by this thread
   pthread_t pt_id = pthread_self();

   // This is the kernel thread id. Each pthread (or user thread) runs on a kernel thread.
   // Multiple user threads may run on a single kernel thread.
   // Now wait at the first barrier.

   #ifndef LINUX
   tid_t kernel_thr_id = thread_self();
   #endif
   barrier1->arrive(tasknum);

   // Stuff this thread's ids in the thread data array

   thread_data_array[next_thread_data_index].pthread_id =  pt_id;
   #ifndef LINUX
   thread_data_array[next_thread_data_index].kernel_thread_id =  kernel_thr_id;
   #endif

   // now wait for the constructor to determine which kernel ids are unique

   barrier1->arrive(tasknum);

   // Now this thread determines whether it will cancel (ie exit) itself.
   // If it does exit, it wont show up at the while loop, to check for work.

   if(thread_data_array[next_thread_data_index].to_cancel) {

      // See Devang Shah's bk, pg 21
      // Detach the thread so it is returned back to the OS and doesnt hog resources

      error =  pthread_detach(pt_id);
      exit_on_error(error, "during thread detach");
      sleep(1);
      pthread_exit((void *) 0);
   }

   // Increment index so that the next thread to be created can use a new slot

   next_thread_data_index++;
   barrier1->arrive(tasknum);

   // First wait to see if there is work.

   status = pthread_mutex_lock(&_queueLock);
   exit_on_error(status," during mutex acquisition ");
   PRINT(pt_id,"entering main loop");

   while(1) {
      PRINT(pt_id,"at beginning of main loop");

      if (++_threadsReady>=_numThreads) {
         PRINT(pt_id,"broadcasting work done");
         status = pthread_cond_broadcast(&_workDone);
         exit_on_error(status," while broadcasting ");
      }

      while(_workStartSent==0) {
         PRINT(pt_id,"waiting on receipt of work start");
         status = pthread_cond_wait(&_workStart, &_queueLock);
         exit_on_error(status," while cond waiting ");
      }
      --_threadsoutstanding;

      if (++_threadsActive==_numThreads) {
         _workStartSent=0;
         PRINT(pt_id,"active and last, broadcasting teamAssembled");
         // status = pthread_cond_broadcast(&_teamAssembled);
         exit_on_error(status," while cond waiting ");
      }

      // Now as long as there is work, keep doing it
      while(1) {
         PRINT(pt_id,"entering work loop");

         // Remove and process a queue item.
         // First check if the queue is empty -- it is possible that other crew members have
         // already started work, and emptied the queue. Note that _workCount is decremented
         // only after a task is successfully completed. Hence it is possible for the queue 
	 // to be empty.

         if(_nextWorkIndex >= _EndOfQueue) {
            PRINT(pt_id,"no work found");
            if (_threadsoutstanding) {
               _threadsReady += _threadsoutstanding;
               _threadsoutstanding =0;
               _workStartSent = 0;
            }
            
	    // 	    while(_threadsActive<_numThreads) {
	    // 	       PRINT(pt_id,"waiting for team");
	    // 	       status = pthread_cond_wait(&_teamAssembled, &_queueLock);
	    // 	       exit_on_error(status," while cond waiting ");
	    // 	    }
            

            PRINT(pt_id,"no work found, team complete");
            break;
         }

         // The next work item, pointed to by _nextWorkIndex will be executed.
         // For now, just print this value in the queue.  Later, call the item in the queue.

         myWorkIndex = _nextWorkIndex;

         // Increment next work index so that the other threads know which QUeue element 
	 // needs processing

         _nextWorkIndex++;
         PRINT(pt_id,"found work, releasing lock to do work");
         status = pthread_mutex_unlock(&_queueLock);
         exit_on_error(status," while unlocking mutex ");

         // Get to work on this item -- basically call that item's execute function

         EXECUTE_TIMINGS_PRINT(pt_id, " starting execute of work unit");
         _QueueRef[myWorkIndex]->execute();
         EXECUTE_TIMINGS_PRINT(pt_id, " finished execute of work unit");

         // Now decrement the work counter after this item is successfully completed, 
	 // and signal if we reach the end of all work

         status = pthread_mutex_lock(&_queueLock);
         exit_on_error(status," during mutex acquisition ");
         _workCount--;
         PRINT(pt_id,"done with work, will look for more");
      }         // of while(1)
   }            // of while (1)
}               // of ThreadPool::worker()

int ThreadPool::processQueue(std::deque<WorkUnit *> & Q_ref)
{
   int status;
   // The first time the Thread Pool is created, _workCount is zero.
   // When the processQueue requestPn is issued, the _workCount is 
   // initialized to the size of the Queue.
   // After all the crew members in the ThreadPool finish their work, 
   // _workCount is set back to zero.
   // This model assumes that only a single thread of master control 
   // exists in the program, e.g. the simulation engine. Multiple 
   // threads are not allowed to call the processQueue request.
   #if defined(_EXECUTE_TIMINGS_PRINT) || defined(_PRINT)
   pthread_t pt_id = pthread_self();
   #endif
   EXECUTE_TIMINGS_PRINT(pt_id, " entering processQueue");
   status = pthread_mutex_lock(&_queueLock);
   if ( report_on_error(status," while acquiring mutex lock, 8 ")) {
      return status;
   }
   PRINT(pt_id," entering processQueue");

   while(_threadsReady<_numThreads) {
      PRINT(pt_id, " waiting for threads to be ready on entry");
      status = pthread_cond_wait(&_workDone, &_queueLock);
      if ( report_on_error(status," while performing cond_wait")) {
         return status;
      }
   }
   // Initialize the internal _QueueRef, which will be used by the worker threads
   _QueueRef =  Q_ref;
   _workCount = Q_ref.size();
   _nextWorkIndex = 0;           // Start work at the front of the queue, at index 0
   // Work from index = 0 to index = (size - 1).  _EndOfQueue is reached when the next 
   // item to be processed (_nextWorkIndex) points to one element beyond the end of the queue.
   _EndOfQueue = _workCount;
   _threadsoutstanding = _numThreads;
   _threadsReady = 0;
   _workStartSent = 1;
   _threadsActive =0;

   PRINT(pt_id, " set up complete, broadcasting work start");
   // Now wake up all the threads which are waiting for this load of work
   status = pthread_cond_broadcast(&_workStart);
   if (report_on_error(status," while performing cond_broadcast")) {
      return status;
   }
   // Now wait until the crew members in the ThreadPool finish this task
   while(_threadsReady<_numThreads) {
      PRINT(pt_id," waiting for threads to be ready after broadcast");
      status = pthread_cond_wait(&_workDone, &_queueLock);
      if (report_on_error(status," while performing cond_wait")) {
         return status;
      }
   }
   // Now the processQueue exits after unlocking the mutex, _queueLock.  
   // Reset the _workCount to zero now
   _workCount = 0;
   PRINT(pt_id," returning");
   status = pthread_mutex_unlock(&_queueLock);
   if (report_on_error(status," while performing unlock")) {
      return status;
   }
   return 0;                     // No errors
}

// An extern "C" function MUST be used as the argument to the thread
// create function (ie a member function cannot be used).
extern "C" void *CWorkerFriend(void *arg)
{
   ThreadPool *ptr_tp = reinterpret_cast<ThreadPool *> (arg);
   return(ptr_tp->worker() );
}

ThreadPool::~ThreadPool()
{
   // Note: without this destructor, you can get weird error messages about mutex_locks 
   // not being acquired when the program terminates.
   // Write the destructor later -- basically free
   // the thread arrays, and terminate the threads.
   int error;
   for(int i = 0; i < _numThreads; i++) {
      // Detach the thread so it is returned back to the OS and doesnt hog resources
      error =  pthread_detach(thread_data_array[i].pthread_id);
      exit_on_error(error,"during thread detach");
   }
   delete[] thread_data_array;
   delete[] thread_ids;
   delete barrier1;
}

void ThreadPool::bindCpus(int NCpus)
{
   // Bind the unique kernel threads to the N cpus. We have created _numThreads
   // unique kernel threads, as requested by the user. If we have _numThreads > NCpus,
   // then multiple threads will be allocated to a CPU, using the "mod" operator (%).
   int status;
   for(int i = 0; i < _numThreads; i++) {
      // Bind threads to CPUS
      status = bindprocessor(BINDTHREAD,  thread_data_array[i].kernel_thread_id, i%NCpus);
      exit_on_error(status,"Failure at bindprocessor");
   }
}
#endif
#endif
