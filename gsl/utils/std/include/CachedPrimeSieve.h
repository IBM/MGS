// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CachedPrimeSieve_H
#define CachedPrimeSieve_H
#include "Copyright.h"

#include <string>
#include <vector>

// This is an implementation of Eratosthenes' sieve algorithm.
// !!! It has a cache for efficiency, so use a mutex before calling functions 
// on this class if multiple threads might be accessing it. Also std::vector
// might cause problems in a MT environment even though GNU people claim 
// access to different containers are safe.
// This algorithm is not very efficient for very large numbers, please use a
// different algorithm for them.
class CachedPrimeSieve
{
   public:
      static void getPrimes(int query, std::vector<int>& primes);
   private:
      static inline void removeDuplicates(
	 int prime, int indexModifier, int candidateSize, 
	 int* candidates);
      static inline int findFirstNonZero(
	 int start, int candidateSize, int* candidates);
      
      static std::vector<int> _cache;
      static int _largestQuery;
};

#endif
