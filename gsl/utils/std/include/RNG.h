// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef RNG_H
#define RNG_H

#ifdef HAVE_GPU

//#if defined(HAVE_GPU) 
//TUAN TODO
//this is better, and we will design a new strategy for using RNG directly on GPU
//The reason is that each instance of RNG needs to keep its own copy of device internal memory
#include "MersenneTwister.h"
typedef MersenneTwister RNG_ns; // non-reseedable
typedef MersenneTwister_S RNG; // reseedable
//#else
//#include "RNG_GPU.h"
//typedef MRG32k3a_GPU RNG_ns; // non-reseedable
//typedef MRG32k3a_S_GPU RNG; // reseedable
//#endif

#else
//#include "MRG32k3a.h"
//typedef MRG32k3a RNG_ns; // non-reseedable
//typedef MRG32k3a_S RNG; // reseedable
#include "MersenneTwister.h"
typedef MersenneTwister RNG_ns; // non-reseedable
typedef MersenneTwister_S RNG; // reseedable
#endif // HAVE_GPU

// Below here is old (as of 08/19/16) code that may not
// work, above will even if switching between the random
// number generator types.

//typedef MersenneTwister RNG;
/*
static RNG Rangen;
static RNG SharedRangen;
static RNG& getRangen() {
      return Rangen;
};
static RNG& getSharedRangen() {
      return SharedRangen;
};
*/
#endif
