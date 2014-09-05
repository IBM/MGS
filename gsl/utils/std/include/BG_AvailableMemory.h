#ifndef BG_AVAILABLEMEMORY
#define BG_AVAILABLEMEMORY

#ifdef USING_BLUEGENEP
#include <sys/resource.h>
#endif
#include <unistd.h>
#include <stdio.h>

// Returns the amount of free memory, in Mbytes

static inline
double AvailableMemory(void)
{
  double MBytes=0;
// Code written by David Latino, IBM
  unsigned long val, st[2];
  st[0] = 123456;
  val = (unsigned long)st;
  MBytes = (double)(val - (unsigned long)sbrk(0)) * 0.00000095367431640625;
#ifdef USING_BLUEGENEP
// Code written by Bob Walkup, IBM
  struct rusage RU;
  getrusage(RUSAGE_SELF, &RU);
  MBytes -= ((double)RU.ru_maxrss)/1024.0;
#endif // USING_BLUEGENEP
  return MBytes;
}
#endif
