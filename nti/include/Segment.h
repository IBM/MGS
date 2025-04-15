// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SEGMENT_H
#define SEGMENT_H

#include <mpi.h>

#include "Sphere.h"
#include "SegmentDescriptor.h"
#include "SegmentForce.h"
#include "../../nti/include/MaxComputeOrder.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <cassert>

#define N_SEG_DATA (N_SPHERE_DATA+3)
#define SEG_ORIG_COORDS N_SPHERE_DATA

#define MIN_MASS 1331.0

class Branch;
class Touch;

class Segment
{
public:
  Segment();
  Segment(Segment const & s);
  Segment& operator=(const Segment& segment);
  double* getVelocity(){return _velocity;}
  double* getForce(){return _force;}
  double getMass(){return _mass;}
  void setKey();
  void setMass(double mass) {_mass=mass;}
  void setFrontLevel(int frontLevel) {_frontLevel = frontLevel;}
  void setDist2Soma(double dist2Soma) {_segmentData._sphere._dist2Soma=dist2Soma;}
  int getFrontLevel() {return _frontLevel;}
  double getDist2Soma() {return _segmentData._sphere._dist2Soma;}
  void setComputeOrder(int computeOrder) {_computeOrder = computeOrder;}
  int getComputeOrder() {return _computeOrder;}
  const int getSegmentIndex() {return _segmentIndex;}
  void setSegmentIndex(int segmentIndex) {_segmentIndex=segmentIndex;}
  const long int getSegmentArrayIndex() {return _segmentArrayIndex;}
  void setSegmentArrayIndex(long int segmentArrayIndex) {_segmentArrayIndex=segmentArrayIndex;}
  Sphere& getSphere() {return _segmentData._sphere;}
  bool isTerminalSegment();

  double* getData() {return _segmentData._data;}
  double* getCoords() {return _segmentData._sphere._coords;}
  double& getRadius() {return _segmentData._sphere._radius;}
  key_size_t getSegmentKey() const {return _segmentData._sphere._key;}
  double* getOrigCoords() {return &_segmentData._data[SEG_ORIG_COORDS];}

  void loadBinary(FILE*, Branch*, const int segmentIndex);
  void loadText(double, double, double, double, Branch*, int);
  void resetBranch(Branch* b);
  void writeCoordinates(FILE*);
  Branch* getBranch() {return _branch;}
  bool isJunctionSegment() {return _isJunctionSegment;}
  void isJunctionSegment(bool p) {_isJunctionSegment=p;}
  void clearForce();
  void addForce(double *F){_force[0] += F[0]; _force[1] += F[1];  _force[2] += F[2];}
  void addTouch(Touch* touch);
  std::list<Touch*>& getTouches() {return _touches;}
  void resetTouches();
  void limitPrecision();

private:
  double _velocity[3];
  double _force[3];
  double _mass;
  int _segmentIndex;
  long int _segmentArrayIndex;
  Branch* _branch;
  bool _isJunctionSegment;
  int _frontLevel;
  int _computeOrder;
  union SegmentData {
    Sphere _sphere;
    double _data[N_SEG_DATA];
  } _segmentData;
  std::list<Touch*> _touches;
  static SegmentDescriptor _segmentDescriptor;
};
#endif

