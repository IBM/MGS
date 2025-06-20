// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H
#define MAXRETURNRANKS 50

#include "Sphere.h"
#include "ShallowArray.h"
#include <mpi.h>

class SegmentSpace;
class TouchSpace;

class Decomposition
{
 public:
  virtual void decompose()=0;
  virtual void getRanks(Sphere* sphere, double* coords2, double deltaRadius, ShallowArray<int, MAXRETURNRANKS, 100>& ranks)=0;
  virtual void addRanks(Sphere* sphere, double* coords2, double deltaRadius, ShallowArray<int, MAXRETURNRANKS, 100>& ranks)=0;
  virtual bool mapsToRank(Sphere* sphere, double* coords2, double deltaRadius, int rank)=0;

  virtual void getRanks(Sphere* sphere, double deltaRadius, ShallowArray<int, MAXRETURNRANKS, 100>& ranks)=0;
  virtual int getRank(Sphere& sphere)=0;
  virtual bool isCoordinatesBased()=0;
  virtual void resetCriteria(SegmentSpace*)=0;
  virtual void resetCriteria(TouchSpace*)=0;
  virtual Decomposition* duplicate()=0;

  virtual void writeToFile(FILE*)=0;
  virtual void readFromFile(FILE*)=0;
  virtual ~Decomposition() {};
};

#endif


