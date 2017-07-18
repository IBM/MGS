// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
