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

/* IMPORTANT: this is necessary for large-scale simulation */
#define ARRAY_LAZY_ALLOCATION

/* as we create CG_LifeNodeGridLayerData 
 * we should not re-create NodeAccessor 
 *  from RuntimePhase
 */
#define REUSE_NODEACCESSORS
//1.--> the REUSE_EXTRACTED_NODESET_FOR_CONNECTION require the activation of REUSE_NODEACCESSORS
//#define REUSE_EXTRACTED_NODESET_FOR_CONNECTION 
//2.--> the TRACK_SUBARRAY_SIZE require the activation of REUSE_NODEACCESSORS
#define TRACK_SUBARRAY_SIZE
//(no longer used)--> the BASED_ON_INDEX require the activation of TRACK_SUBARRAY_SIZE 
     //#define BASED_ON_INDEX

#define OLD_APPROACH 0
#define NEW_APPROACH 1
//#define DETECT_PROXY_COUNT NEW_APPROACH   // NOT COMPLETED
//      ... BETTER
#define DETECT_NODE_COUNT NEW_APPROACH 

#define SUPPORT_MULTITHREAD_NODEACCESSOR_SETUP 
/*
 * in CG_xxxGridLayerData.C 
 * the setup of _nodeInstanceAccessor[n] is as serial process
 * which becomes too slow on extremely large grid
 *
 */

#define USE_ONLY_MAIN_THREAD 0
#define USE_DYNAMIC_MULTIPLE_THREAD 1
#define USE_STATIC_THREADPOOL 2
// IF defined (in any form), it means we use a vector to tracks (dest)-nodes that the current source-node 
// .. want to connect; or tracks (source)-nodes that connect to the current dest-node
//#define SUPPORT_MULTITHREAD_CONNECTION USE_ONLY_MAIN_THREAD
//DON'T USE the below for now, as not working for large-scale problem due to dynamic thread creation
//  ... and even using ThreadPool - maybe the number of threads taking resources -> slower
//#define SUPPORT_MULTITHREAD_CONNECTION USE_DYNAMIC_MULTIPLE_THREAD
//#define SUPPORT_MULTITHREAD_CONNECTION USE_STATIC_THREADPOOL
//   test the idea that only split the destsets 
//     for each node in sourceset, do 
//        parallel-detect nodes in the destsets  [save quite amount of time]
//   ----
//   then sequential connect node in sourceset, to selected nodes in destsets 

// we need this to make MULTITHREAD_CONNECTION work on large problem
//   ... to avoid creating/destroying threads repeatedly
#if defined(SUPPORT_MULTITHREAD_CONNECTION) && SUPPORT_MULTITHREAD_CONNECTION == USE_STATIC_THREADPOOL
#define USE_THREADPOOL_C11
#endif
#if defined(SUPPORT_MULTITHREAD_CONNECTION) && \
  (SUPPORT_MULTITHREAD_CONNECTION == USE_DYNAMIC_THREADPOOL || \
  SUPPORT_MULTITHREAD_CONNECTION == USE_STATIC_THREADPOOL) 
#define USING_SUB_NODESET
#endif

/* NOT COMPLETED */
//#define TEST_MULTIPLE_THREADS  //test the idea of splitting both sourceset and destsets into K subgroups
//   and do sourceset[i] --> sourceset[j]

/* NOT COMPLETED */
#define USE_STD_ALLOCATOR  0
#define USE_PLACEMENT_NEW  1
#define FLAT_MEM_MANAGEMENT USE_STD_ALLOCATOR
//#define FLAT_MEM_MANAGEMENT  USE_PLACEMENT_NEW

/* track for fast searching */
#define TEST_IDEA_TRACK_GRANULE

/* this enable add all nodes in parallel 
 * also, allocateNodes()
 *   does 2 things: allocate memory, constructors, and set the size
 *   to the desired value
 * NOTE: need 'TEST_IDEA_TRACK_GRANULE' enabled
 * */
#define PARALLEL_ADD_NODE

#if defined(SUPPORT_MULTITHREAD_CONNECTION) || \
    defined(SUPPORT_MULTITHREAD_NODEACCESSOR_SETUP) || \
    defined(PARALLEL_ADD_NODE)
#include <thread>
#include <mutex>
#include <functional>
#include <algorithm>
#endif

#define OPTION_3 3
#define OPTION_4 4
#define OPTION_4b 41
#define OPTION_5 5

//NOTE: for proxy
//  OPTION_3 = means each CCDemarshaller has its own array of data, tracking the mapping to nodes in a different rank -- BETTER
//  OPTION_4 = means each CCDemarshaller only keep the index to the data, 
//    data nows belong to the global proxy-data array in the xxxCompCategory
#define PROXY_ALLOCATION  OPTION_3
//#define PROXY_ALLOCATION  OPTION_4

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
// OPTION_4 means  (we need three arrays to track with MAX_SUBARRAY_SIZE is used)
//         [potentially to behave like OPTION_5 if we replace MAX_SUBARRAY_SIZE with the exact-size of subarray]
//   class CG_LifeNodeCompCategory{
//   ShallowArray_Flat< float*>  data;
//   ShallowArray_Flat< int>  data_start_offset;
//   ShallowArray_Flat< int>  data_num_elements;
//   }
// OPTION_4b means  (we need 2 arrays to track with MAX_SUBARRAY_SIZE is used)
//   class CG_LifeNodeCompCategory{
//   ShallowArray_Flat< float*>  data;
//   ShallowArray_Flat< int>  data_num_elements;
//   int data_max_elements; // = MAX_SUBARRAY_SIZE
//   }
// OPTION_5 means  (we need 2 arrays to track if we know exactly how many elements for each subarray)
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
//   support: OPTION_3, OPTION_4, OPTION_4b
//   non-complete: OPTION_5 (no data declaration for OPTION_5 and is similar to OPTION_4 with turning off USE_SHARED_MAX_SUBARRAY)
//   Use OPTION_3 if we want the sub-array to be resized by any InitPhase functions
#define DATAMEMBER_ARRAY_ALLOCATION OPTION_3
//#define DATAMEMBER_ARRAY_ALLOCATION OPTION_4b

// IF (1), then the size information is stored at 
//  std::pair(-1, -1)
#define USE_SHARED_MAX_SUBARRAY 1

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


/* x value range -2 to 2
 * y value: 0 to 1
 * small change in x brings large change in y
 * USE: usually used in output layer of a binary classification [result is 0 or 1]
 * NOTE: Use softmax for multiple-class classification
 */
template<typename T>
CUDA_CALLABLE T sigmoid(T x)
{
  return exp(x) / (1 + exp(x));
}

/* Rectified Linear Unit
 * y in range [0, inf]
 * USE: most widely used activation function [in hidden layers]
 *  as we can easily calculate the differential (less computational expensive than 'tanh', 'sigmoid' function), 
 *  --> we can easily do backpropagate errors
 *  also, at a time, only a few neurons are activated -> making the network sparse, i.e. efficient and easy for computation
 *
 */
template<typename T>
CUDA_CALLABLE T ReLU(T x)
{
  return fmaxf(x, 0);
}

/* 
- a shifted version of sigmoid
 * y value: -1 to 1
 * USE: usually used in hidden layer 
 * [the means of the layer is closed to 0, helps in centering the data 
 *    -> make learning for next layer much easier]
 * almost always work better than sigmoid
*/
template<typename T>
CUDA_CALLABLE T tanh(T x)
{
  return 2/(1+exp(-2*x))-1; 
}


#endif
