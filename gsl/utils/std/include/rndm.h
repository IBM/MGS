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

#ifndef rndm_h
#define rndm_h

#include <limits.h>
#include <math.h>
#include <cassert>

#include "RNG.h"
#if defined(HAVE_GPU) && defined(__NVCC__)
#define CUDA_CALLABLE __host__ __device__
#define TEST_USING_GPU_COMPUTING
#ifndef CUDA_CHECK_CODE
#define CUDA_CHECK_CODE
#define gpuErrorCheck(r) {_check((r), __FILE__, __LINE__);}
inline    void _check(cudaError_t r, const char* file, int line, bool abort=true) 
{
  if (r != cudaSuccess) {
    printf("CUDA error in file %s on line %d: %s --> %s\n", file, line, cudaGetErrorName(r), cudaGetErrorString(r));
    if (abort) exit(0);
  }
}

#define OPTION_3 3
#define OPTION_4 4

#define PROXY_ALLOCATION  OPTION_3
//#define PROXY_ALLOCATION  OPTION_4

#endif
#else
#define CUDA_CALLABLE
#endif

inline double drandom(RNG_ns& rangen)
{
   return rangen.drandom32();
}

inline double drandom(double min, double max, RNG_ns& rangen)
{
   return ((max-min)*drandom(rangen) + min);
}

inline long lrandom(RNG_ns& rangen)
{
   return rangen.irandom32();
}

inline long lrandom(long min, long max, RNG_ns& rangen)
{
   return long(floor(drandom(double(min) ,double(max+1), rangen)));
}

inline int irandom(RNG_ns& rangen)
{
   return int(rangen.drandom32()* INT_MAX);
}

inline int irandom(int min, int max, RNG_ns& rangen)
{
  return int(floor(drandom(double(min), double(max+1), rangen)));
}

inline int urandom(RNG_ns& rangen)
{
   return unsigned(rangen.drandom32()* UINT_MAX);
}

inline double gaussian(RNG_ns& rangen)
{
   static int gaussian_flag = 0;
   static double gaussian_deviate;
   double fac, r, v1, v2;

   if (gaussian_flag == 0) {
      do {
         v1 = drandom(-1.0, 1.0, rangen);
         v2 = drandom(-1.0, 1.0, rangen);
         r = v1*v1 + v2*v2;
      }while (r >= 1.0  || r == 0.0);
      fac = sqrt(-2.0 * log(r)/r);
      gaussian_deviate = v1 * fac;
      gaussian_flag  = 1;
      return v2*fac;
   }
   else {
      gaussian_flag =0;
      return gaussian_deviate;
   }
}

inline double expondev (RNG_ns& rangen)
{
   double tmp;
   do {
      tmp = drandom(rangen);
   }while(tmp == 0.0);
   return -log(tmp);
}

inline double expondev (double g, RNG_ns& rangen)
{
   return expondev(rangen) /g;
}

inline double gaussian(double mean, double sd, RNG_ns& rangen)
{
   return (sd*gaussian(rangen) + mean);
}

#endif
