// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
