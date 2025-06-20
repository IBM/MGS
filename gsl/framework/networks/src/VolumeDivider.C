// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "VolumeDivider.h"
#include "CachedPrimeSieve.h"
#include "VolumeOdometer.h"
#include "NodeDescriptor.h"
#include "ConnectionIncrement.h"
#include "NodeSet.h"
#include "Simulation.h"
#include "DistributableCompCategoryBase.h"
#include "DataItem.h"
#include "CustomStringDataItem.h"
#include "ArrayDataItem.h"
#include "IntArrayDataItem.h"
#include "NumericDataItem.h"
#include "FunctorDataItem.h"
#include "LayoutFunctor.h"
#include "Simulation.h"
#include "VectorOstream.h"

#include <iostream>
#include <cassert>
#include <algorithm>

//#define INSTRUMENT

VolumeDivider::VolumeDivider()
   : _simpleMethod(false), _numPieces(0)
{
}

void VolumeDivider::setUp(std::vector<int>& dimensions, unsigned numGranules)
{
  assert(dimensions.size() > 0);
  _dimensions=_minChunkSizes=dimensions;
  _numPieces=numGranules;
   
  std::vector<int>::const_iterator dit, dend = dimensions.end();
  for (dit = dimensions.begin(); dit != dend; ++dit) {
    assert(*dit > 0);
  }
  for (unsigned i = 0; i < dimensions.size(); ++i) {
    _dividers.push_back(1);
  }

  findPieces();
   
  int totalSize = 1;
  std::vector<int>::iterator it, end = dimensions.end();
  for (it = dimensions.begin(); it != end; ++it) {
    totalSize *= (*it);
  }
   
  if (_numPieces > totalSize) {
    _simpleMethod = true;
  }
   
  if (!_simpleMethod) {
    for (unsigned i = 0; i < dimensions.size(); ++i) {
      _remainders.push_back(dimensions[i] - (_minChunkSizes[i] * _dividers[i]));
    }
     
    for (unsigned i = 0; i < dimensions.size(); ++i) {
      //_cutOffs.push_back((_minChunkSizes[i] + 1) * _remainders[i]);
      _cutOffs.push_back(_minChunkSizes[i] * (_dividers[i]-_remainders[i]));
    }
    _strides.resize(dimensions.size());
    _strides[_strides.size() - 1] = 1;
     
    if (_strides.size() > 1) {
      for (int i = dimensions.size() - 2; i >= 0; --i) {
	_strides[i] = _strides[i + 1] * _dividers[i + 1];
      }
    }
  }

  
  // compute the dimOrder vector, which for each dimension provides the rank (zero=minimum) of the number of dividers for that dimension
  _dimOrder.resize(_dimensions.size(), -1);
  unsigned count=0;
  while (count<dimensions.size()) {
    int min=INT_MAX;
    int minIdx=-1;
    for (unsigned i = 0; i < dimensions.size(); ++i) {
      if (_dividers[i]<min) {
	if (_dimOrder[i]>=0) continue;
	min=_dividers[i];
	minIdx=i;
      }
    }
    assert(minIdx>=0);
    _dimOrder[minIdx]=count;
    ++count;
  }  
#ifdef INSTRUMENT
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank==0) { 
    std::cerr<<"minChunkSizes = "<<_minChunkSizes<<std::endl<<"dividers = "<<_dividers<<std::endl<<"cutOffs = "<<_cutOffs<<std::endl<<"remainders = "
	     <<_remainders<<std::endl<<"dimensions = "<<dimensions<<std::endl<<"strides = "<<_strides<<std::endl<<"dimOrder = "<<_dimOrder<<std::endl;
  }
#endif
}

unsigned VolumeDivider::getPiece(const std::vector<int>& coordinates) const
{
   unsigned partition = 0;

   if (_simpleMethod) {
      partition = 0;
   } else {
      for (unsigned i = 0; i < _strides.size(); ++i) {
         partition += getSubPiece(i, coordinates[i]) * _strides[i];
      }   
   }
   return partition;
}

void VolumeDivider::getPieceCoordinates(unsigned partition, std::vector<double>& coordinates)
{
  if (_simpleMethod) coordinates.resize(_minChunkSizes.size(),0);
  else {
    coordinates.resize(_dimensions.size());
    for (unsigned i = 0; i < _dimensions.size(); ++i) {
      unsigned subPiece = partition/_strides[i];
      partition=partition%_strides[i];      
      // This method assigns the least "corner" coordinate to the granule
      if (_dimOrder[i]>int(i)) coordinates[_dimOrder[i]]=double(_dividers[i]-subPiece-1) / double(_dividers[i]);
      else coordinates[_dimOrder[i]]=double(subPiece) / double(_dividers[i]);
    }
  }
}


unsigned VolumeDivider::getSubPiece(int dimension, int coordinate) const
{
   unsigned subPartition;
   if (coordinate < _cutOffs[dimension]) {
      // These have minSize + 1
      subPartition = coordinate / _minChunkSizes[dimension];
   } else {
      subPartition = _dividers[dimension] - _remainders[dimension] + 
	(coordinate-_cutOffs[dimension])/(_minChunkSizes[dimension]+1) ;
   }
   return subPartition;
}

void VolumeDivider::findPieces()
{
   std::deque<int> factors;
   decompose(_numPieces, factors);
   std::deque<int>::iterator it, end = factors.end();
   int maxIndex;
   for (it = factors.begin(); it != end; ++it) {
      maxIndex = indexMaxAvgChunk();
      _dividers[maxIndex] *= *it;
      _minChunkSizes[maxIndex] /= *it;
   }
}

void VolumeDivider::decompose(int number, std::deque<int>& factors) const
{
  factors.clear();
  std::vector<int> primes;
  CachedPrimeSieve::getPrimes(number, primes);
  std::vector<int>::iterator it, end = primes.end();
  for (it = primes.begin(); it != end; ++it) {
    while ((number % *it) == 0) {
      factors.push_front(*it);
      number /= *it;
    }
    if (number == 1) {
      break;
    }
  }
  assert(number == 1);
}

int VolumeDivider::indexMaxAvgChunk() const
{
   int maxDimension = 0;  
   int index = 0;
   
   for (unsigned i = 0; i < _minChunkSizes.size(); ++i) {
     if (_minChunkSizes[i] > maxDimension) {
       maxDimension = _minChunkSizes[i];
       index = i;
     }
   }
   return index;
}

VolumeDivider::~VolumeDivider()
{
}
