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

#ifndef CAPSULE_H
#define CAPSULE_H
#include <mpi.h>

#include "Sphere.h"
#include "Touch.h"
#include "SegmentDescriptor.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <cassert>
#include <string.h>

#define N_CAP_DATA (N_SPHERE_DATA+3)
#define CAP_END_COORD N_SPHERE_DATA

class ComputeBranch;

class Capsule
{
public:
  Capsule();
  Capsule(Capsule const & c);
  ~Capsule();
  bool operator<(const Capsule& c1) const;
  bool operator==(const Capsule& c1) const;
 
  double* getData() {return _capsuleData._data;}
  double* getBeginCoordinates() {return _capsuleData._sphere._coords;}
  double getRadius() {return _capsuleData._sphere._radius;}
  void setRadius(double radius) {_capsuleData._sphere._radius=radius;}
  double getKey() const {return _capsuleData._sphere._key;}
  double getDist2Soma() const {return _capsuleData._sphere._dist2Soma;}
  int getSurfaceArea();
  void setKey(double key) {_capsuleData._sphere._key=key;}
  void setDist2Soma(double dist2Soma) {_capsuleData._sphere._dist2Soma=dist2Soma;}
  double* getEndCoordinates() {return &_capsuleData._data[CAP_END_COORD];}
  Sphere& getSphere() {return _capsuleData._sphere;}
  void getEndSphere(Sphere& sphere);
  int getEndSphereSurfaceArea();
  ComputeBranch* getBranch() {return _branch;}
  void setBranch(ComputeBranch* branch) {_branch=branch;}
  double getEndProp();
  void readFromFile(FILE* dataFile);
  void writeToFile(FILE* dataFile);

private:
  union CapsuleData {
    Sphere _sphere;
    double _data[N_CAP_DATA];
  } _capsuleData;
  ComputeBranch* _branch;
  static SegmentDescriptor _segmentDescriptor;
};
#endif

