// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

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
};

#endif


