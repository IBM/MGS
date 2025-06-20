// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
// Implements a barrier for synchronizing multiple
// threads. Wait until all threads finish before continuing.
#include "Copyright.h"

#ifndef _BARRIER_H
#define _BARRIER_H

#ifndef DISABLE_PTHREADS
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>


class Barrier
{
   public:
      Barrier(int n);
      ~Barrier();

      void arrive (int &i);

      void go ();                // raise the barrier
      void wait ();              // wait until all member have reached
      // the barrier

   private:

      unsigned int nb_waiters;
      unsigned total_threads;
      pthread_cond_t local_control;
      pthread_cond_t external_control;
      pthread_mutex_t _mutex;    // mutex to protect changes to 'state' variable

      void askToPass();

};

#include "Barrier.inline.h"
#endif                           // DISABLE_PTHREADS
#endif                           //_BARRIER_H
