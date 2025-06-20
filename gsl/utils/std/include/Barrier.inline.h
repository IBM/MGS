// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _BARRIER_INLINE_H
#define _BARRIER_INLINE_H
#ifndef DISABLE_PTHREADS
#include <pthread.h>
#include "Copyright.h"

#include "Barrier.h"

inline
void Barrier::askToPass ()
{
   if (nb_waiters >= total_threads) {
      pthread_cond_broadcast(&external_control);
   }

   // Note: we don't need
   // a "pthread_mutex_lock(&_mutex)"  here
   // as it has already been done before calling askToPass()

   pthread_cond_wait(&local_control, &_mutex);

}


inline
Barrier::Barrier (int n):
total_threads(n)
{

   nb_waiters = 0;
   pthread_mutex_init(&_mutex, NULL);
   pthread_cond_init(&external_control, NULL);
   pthread_cond_init(&local_control, NULL);
}


inline
Barrier::~Barrier ()
{

}


inline
void Barrier::arrive (int &arrival_number)
{

   pthread_mutex_lock(&_mutex);

   arrival_number = nb_waiters;  // Return arrival number here (e.g. 0th, 1st, 2nd etc)
   // Note: this number starts at 0
   nb_waiters++;

   askToPass();

   pthread_mutex_unlock(&_mutex);

}


inline
void Barrier::go ()
{
   pthread_mutex_lock(&_mutex);

   // open the barrier
   pthread_cond_broadcast(&local_control);

   nb_waiters = 0;
   pthread_mutex_unlock(&_mutex);

}


inline
void Barrier::wait ()
{
   pthread_mutex_lock(&_mutex);
   while (nb_waiters < total_threads) {
      pthread_cond_wait(&external_control, &_mutex);
   }
   pthread_mutex_unlock(&_mutex);

}
#endif                           // DISABLE_PTHREADS
#endif                           //_BARRIER_INLINE_H
