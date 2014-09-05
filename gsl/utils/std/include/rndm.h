#ifndef rndm_h
#define rndm_h

#include <limits.h>
#include <math.h>
#include <cassert>

#include "RNG.h"

inline double drandom(RNG& rangen=getRangen())
{
   return rangen.drandom32();
}

inline double drandom(double min, double max, RNG& rangen=getRangen())
{
   return ((max-min)*drandom(rangen) + min);
}

inline long lrandom(RNG& rangen=getRangen())
{
   return rangen.irandom32();
}

inline long lrandom(long min, long max, RNG& rangen=getRangen())
{
   return long(floor(drandom(double(min) ,double(max+1), rangen)));
}

inline int irandom(RNG& rangen=getRangen())
{
   return int(rangen.drandom32()* INT_MAX);
}

inline int irandom(int min, int max, RNG& rangen=getRangen())
{
  return int(floor(drandom(double(min), double(max+1), rangen)));
}

inline double gaussian(RNG& rangen=getRangen())
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

inline double expondev (RNG& rangen=getRangen())
{
   double tmp;
   do {
      tmp = drandom(rangen);
   }while(tmp == 0.0);
   return -log(tmp);
}

inline double expondev (double g, RNG& rangen=getRangen())
{
   return expondev(rangen) /g;
}

inline double gaussian(double mean, double sd, RNG& rangen=getRangen())
{
   return (sd*gaussian(rangen) + mean);
}

#endif
