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
