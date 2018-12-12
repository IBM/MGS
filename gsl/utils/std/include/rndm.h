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
#if defined(HAVE_GPU) 

#include "driver_types.h"
#include <cuda_runtime.h>

#define CUDA_CALLABLE __host__ __device__

#define TEST_USING_GPU_COMPUTING

//#define TEST_INITNODE_ON_CPU

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
#define OPTION_4b 41
#define OPTION_5 5

//NOTE: for proxy
//  OPTION_3 = means each CCDemarshaller has its own array of data, tracking the mapping to nodes in a different rank
//  OPTION_4 = means each CCDemarshaller only keep the index to the data, 
//    data nows belong to the global proxy-data array in the xxxCompCategory
//#define PROXY_ALLOCATION  OPTION_3
#define PROXY_ALLOCATION  OPTION_4

//NOTE: By default, a scalar data member (say float type) of LifeNode is re-organized as
//   class CG_LifeNodeCompCategory{
//   //float*  data;
//   ShalloaArray_Flat<float> data;
//   }
// Howabout an array data member (say int*[]) of LifeNode
// HERE
// OPTION_3 means 
//   class CG_LifeNodeCompCategory{
//   ShallowArray_Flat< ShallowArray_Flat<float*>>  data;
//   }
// OPTION_4 means  (we need three array to track with MAX_SUBARRAY_SIZE is used)
//   class CG_LifeNodeCompCategory{
//   ShallowArray_Flat< float*>  data;
//   ShallowArray_Flat< int>  data_start_offset;
//   ShallowArray_Flat< int>  data_num_elements;
//   }
// OPTION_4b means  (we need 2 array to track with MAX_SUBARRAY_SIZE is used)
//   class CG_LifeNodeCompCategory{
//   ShallowArray_Flat< float*>  data;
//   ShallowArray_Flat< int>  data_num_elements;
//   int data_max_elements; // = MAX_SUBARRAY_SIZE
//   }
// OPTION_5 means  (we need two array to track if we know exactly how many elements for each subarray)
//   and the number of elements for 'i'-th node is 
//                     (data_start_offset[i+1] - data_start_offset[i])     if 0<i<n-1
//                     (data_start_offset.size() - data_start_offset[i])   if i = n-1
//         with 'n' is number of nodes
//   class CG_LifeNodeCompCategory{
//   ShallowArray_Flat< float*>  data;
//   ShallowArray_Flat< int>  data_start_offset;
//   }
//
//   IMPORTANT:  for OPTION_4 and OPTION_5, we need to revise global kernel behavior
//   support: OPTION_3, OPTION_4, OPTION_4b, OPTION_5
//#define DATAMEMBER_ARRAY_ALLOCATION OPTION_3
#define DATAMEMBER_ARRAY_ALLOCATION OPTION_4b

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
