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

#include "CachedPrimeSieve.h"
#include <cassert>

std::vector<int> CachedPrimeSieve::_cache;
int CachedPrimeSieve::_largestQuery = 0;

void CachedPrimeSieve::getPrimes(int query, std::vector<int>& primes)
{
   primes.clear();
   // No primes < 2, if you remove this fix "query - indexModifier" below
   if (query < 2) {
      return; 
   }
   if (query > _largestQuery) {
      int indexModifier;

      // Special case for beginning
      if (_largestQuery < 2) {
	 indexModifier = 2;
      } else {
	 indexModifier = _largestQuery + 1;
      }
      
      int candidateSize = (query - indexModifier) + 1;
      int* candidates = new int[candidateSize];
      for (int i = 0; i < candidateSize; ++i) {
	 candidates[i] = 1;
      }
      
      // Remove duplicates for current cache elements
      std::vector<int>::iterator it, end = _cache.end();
      for (it = _cache.begin(); it != end; ++it) {
	 removeDuplicates(*it, indexModifier, candidateSize, candidates);
      }
      int prime = 0;
      while ((prime = findFirstNonZero(prime, candidateSize, candidates)) 
	     != -1) {
	 _cache.push_back(prime + indexModifier);
	 removeDuplicates(prime + indexModifier, indexModifier, 
			  candidateSize, candidates);
      }
      delete[] candidates;
      _largestQuery = query;
   }
   
   std::vector<int>::iterator it, end = _cache.end();
   for (it = _cache.begin(); it != end; ++it) {
      if (*it > query) {
	 break;
      }
   }
   primes.insert(primes.end(), _cache.begin(), it);
}

void CachedPrimeSieve::removeDuplicates(
   int prime, int indexModifier, int candidateSize, int* candidates)
{
   int it; 
   int difference = indexModifier % prime;
   if (difference == 0) {
      it = indexModifier;
   } else {
      it = indexModifier - difference + prime;
   }
   int end = candidateSize + indexModifier;
   for (; it < end; it += prime) {
      candidates[it - indexModifier] = 0;      
   }   
}

int CachedPrimeSieve::findFirstNonZero(int start, int candidateSize, 
				       int* candidates) 
{
   for (int i = start; i < candidateSize; ++i) {
      if (candidates[i] == 1) {
	 return i;
      }
   }
   return -1;
}
