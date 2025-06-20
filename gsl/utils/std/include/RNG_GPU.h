// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef RNG_GPU_H
#define RNG_GPU_H

#ifdef HAVE_GPU

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <math.h>
#include <stdio.h>
#include <iostream>

#define GPU_ERR(err) {checkGPUerror((err), __FILE__, __LINE__);}
#define GPU_RS(err) {checkGPUrandStatus((err), __FILE__, __LINE__);}

static inline void checkGPUerror(cudaError_t err, const char* f, int ln){
  if (err!=cudaSuccess) {
    std::cerr << "In file " << f << "line " << ln <<  ": error type " << cudaGetErrorString(err) << std::endl;
    exit(0);
  }
}

static inline void checkGPUrandStatus(curandStatus_t err, const char* f, int ln){
  if (err!=CURAND_STATUS_SUCCESS) {
    std::cerr << "In file " << f << "line " << ln <<  ": error type unknown related to curand_status" << std::endl;
    exit(0);
  }
}

// Non-reseedable version
class MRG32k3a_GPU
{
 protected:
  size_t N; // size of random number arrays
  size_t i; // current position in array
  curandGenerator_t gen; // generator object
  double* host; // host version
  double* dev; // device version
  
 public:
  MRG32k3a_GPU()
    {
      N = 320000; // multiples of 32 probably most efficient
      i = 0;
      // Host memory
      host = (double*) calloc(N, sizeof(double));
      // Device memory
      GPU_ERR( cudaMalloc((void**) &dev, N*sizeof(double)) );
      // Generator object
      GPU_RS( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A) );
      // Seed it
      GPU_RS( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
      GPU_RS( curandSetGeneratorOffset(gen, 0) );    
      // Generate first set of numbers
      GPU_RS( curandGenerateUniformDouble(gen, dev, N) );
      // Copy to host
      GPU_ERR( cudaMemcpy(host, dev, N*sizeof(double), cudaMemcpyDeviceToHost) );
    }

  ~MRG32k3a_GPU()
    {
      // Generator object
      GPU_RS( curandDestroyGenerator(gen) );
      // Device memory
      GPU_ERR( cudaFree(dev) );
      // Host memory
      free(host);
    }

  inline double drandom32(void)
  {
    // Get next number
    double x = host[i];
    // If at the end generate a new set and copy to host
    if (i++ >= N)
      {
        i = 0;
        GPU_RS( curandGenerateUniformDouble(gen, dev, N) );
        GPU_ERR( cudaMemcpy(host, dev, N*sizeof(double), cudaMemcpyDeviceToHost) );
      }
    return x;
  }

  inline unsigned long irandom32(void)
  {
    return long(floor( LONG_MAX * drandom32() ) );
  }
};

// Reseedable version
class MRG32k3a_S_GPU: public MRG32k3a_GPU
{
 public:

  void reSeed(unsigned seed, unsigned rank)
  {
    seed+=rank;
    reSeedShared(seed);    
  }

  void reSeedShared(unsigned seed) 
  {
    // Seed it
    GPU_RS( curandSetPseudoRandomGeneratorSeed(gen, seed) );
    GPU_RS( curandSetGeneratorOffset(gen, 0) );    
    // Generate new set of numbers
    GPU_RS( curandGenerateUniformDouble(gen, dev, N) );
    GPU_ERR( cudaMemcpy(host, dev, N*sizeof(double), cudaMemcpyDeviceToHost) );
    // Reset i so is deterministic
    i = 0;
  }
};

#endif // HAVE_GPU

#endif // RNG_GPU_H

