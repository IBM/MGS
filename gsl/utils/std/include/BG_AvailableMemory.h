#ifndef BG_AVAILABLEMEMORY
#define BG_AVAILABLEMEMORY
/* =================================================================
Licensed Materials - Property of IBM

"Restricted Materials of IBM"

BMC-YKT-07-18-2017

(C) Copyright IBM Corp. 2005-2017  All rights reserved

US Government Users Restricted Rights -
Use, duplication or disclosure restricted by
GSA ADP Schedule Contract with IBM Corp.

=================================================================
*/

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
