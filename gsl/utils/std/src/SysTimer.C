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

// SysTimer.cpp
//
// Ravi Rao, Sept 15, 2001
//
// This file contains the implementation for the class SysTimer which
// is a stopwatch for the system clock as specified in SysTimer.h.
//
// This implementation uses the C library time.h

#include "SysTimer.h"
#include <time.h>

clock_t myStartTime;             // holds the last time the watch was started
// added Ravi Rao
struct timeval myStartTimeval;

struct timeval myElapsedTimeval;

SysTimer::SysTimer()
: amRunning(false),
myElapsed(0),
myCumulative(0)
{
   myStartTimeval.tv_sec = 0;
   myStartTimeval.tv_usec = 0;
   myElapsedTimeval.tv_sec = 0;
   myElapsedTimeval.tv_usec = 0;
   myCumulativeTimeval.tv_sec = 0;
   myCumulativeTimeval.tv_usec = 0;

}


void SysTimer::start()
// precondition: stopwatch is not running
// postcondition: stopwatch is running
{
   if (amRunning) {
      std::cerr << "attempt to start an already running stopwatch" << std::endl;
   }
   else {
      amRunning = true;
      myStartTime = clock();
      gettimeofday(&myStartTimeval, NULL);
   }
}


void SysTimer::stop()
// precondition: stopwatch is running
// postcondition: stopwatch is stopped
{
   if (! amRunning) {
      std::cerr << "attempt to stop a non-running stopwatch" << std::endl;
   }
   else {
      clock_t myEndTime = clock();
      struct timeval myEndTimeval;
      gettimeofday(&myEndTimeval, NULL);
      // set the status variables
      myElapsed = myEndTime - myStartTime;
      myCumulative += myElapsed;

      timevalSubtract(myElapsedTimeval, myEndTimeval, myStartTimeval);
      timevalAdd(myCumulativeTimeval, myCumulativeTimeval, myElapsedTimeval);

      amRunning = false;         // turn off the stopwatch
   }
}


void SysTimer::reset()
// postcondition: all history of stopwatch use erased
//                and the stopwatch is stopped
{

   amRunning = false;
   myElapsed = 0;
   myCumulative = 0;
   myElapsedTimeval.tv_sec =  myElapsedTimeval.tv_usec = 0;
   myCumulativeTimeval.tv_sec  = myCumulativeTimeval.tv_usec = 0;
}


bool SysTimer::isRunning() const
// postcondition: returns true iff stopwatch is currently running
{
   return amRunning;
}


double SysTimer::lapTime() const
// precondition: stopwatch is running
// postcondition: returns time in microseconds since last start
{
   return amRunning ?
      (double)(clock() - myStartTime) / CLOCKS_PER_SEC : 0.0;
}


double SysTimer::lapWallTime()
// precondition: stopwatch is running
// postcondition: returns time in microseconds since last start
{

   struct timeval temp, temp2;
   gettimeofday(&temp, NULL);
   timevalSubtract(temp2, temp, myStartTimeval);

   double result;
   result = temp2.tv_sec + temp2.tv_usec/1000000.0;

   if( amRunning)
      return(result);
   else
      return(0);

}


double SysTimer::elapsedTime() const
// precondition: stopwatch is not running
// postcondition: returns time in microseconds between last stop and start
{
   return amRunning ? 0.0 : (double)myElapsed / CLOCKS_PER_SEC;
}


double SysTimer::elapsedWallTime() const
// precondition: stopwatch is not running
// postcondition: returns time in microseconds between last stop and start
{
   return amRunning ? 0.0 : (double)myElapsedTimeval.tv_sec + (myElapsedTimeval.tv_usec/1000000.0);
}


double SysTimer::cumulativeTime() const
// postcondition: returns total time stopwatch has been active
{
   return ((double)myCumulative / CLOCKS_PER_SEC) + lapTime();
}


double SysTimer::cumulativeWallTime()
// postcondition: returns total time stopwatch has been active
{
   double result;

   result = myCumulativeTimeval.tv_sec + myCumulativeTimeval.tv_usec/1000000.0;
   result += lapWallTime();
   return result;
}


double SysTimer::granularity() const
{
   return 1.0 / CLOCKS_PER_SEC;
}


void   SysTimer::timevalSubtract(struct timeval &result, struct timeval &time1, struct timeval &time2)

{

   if ((time1.tv_sec < time2.tv_sec) ||
      ((time1.tv_sec == time2.tv_sec) &&
      /* TIME1 <= TIME2? */
   (time1.tv_usec <= time2.tv_usec))) {
      result.tv_sec = result.tv_usec = 0 ;
   }                             /* TIME1 > TIME2 */
   else {
      result.tv_sec = time1.tv_sec - time2.tv_sec ;
      if (time1.tv_usec < time2.tv_usec) {
         result.tv_usec = time1.tv_usec + 1000000 - time2.tv_usec ;
         result.tv_sec-- ;       /* Borrow a second. */
      }
      else {
         result.tv_usec = time1.tv_usec - time2.tv_usec ;
      }
   }
}


void   SysTimer::timevalAdd(struct timeval &result,
struct timeval &time1, struct timeval &time2)

{

   result.tv_sec = time1.tv_sec + time2.tv_sec ;
   result.tv_usec = time1.tv_usec + time2.tv_usec ;
                                 /* Carry? */
   if (result.tv_usec > 1000000) {
      result.tv_sec++ ;  result.tv_usec = result.tv_usec - 1000000 ;
   }

}
