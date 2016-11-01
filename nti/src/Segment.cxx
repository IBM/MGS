// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. and EPFL 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#include "Segment.h"
#include "Branch.h"
#include "VecPrim.h"

SegmentDescriptor Segment::_segmentDescriptor;

Segment::Segment()
  : _mass(0), _segmentIndex(0), _segmentArrayIndex(0), _branch(0), _isJunctionSegment(false), _frontLevel(-1), _computeOrder(0)
{
  for (int i=0; i<3; ++i) {
    _velocity[i]=0;
    _force[i]=0;
  }
  for (int i=0; i<N_SEG_DATA; ++i) {
    _segmentData._data[i]=0;
  }
}

Segment::Segment(Segment const & s)
  : _mass(s._mass), _segmentIndex(s._segmentIndex), _segmentArrayIndex(s._segmentArrayIndex), 
    _branch(s._branch), _isJunctionSegment(s._isJunctionSegment), _frontLevel(s._frontLevel),
    _computeOrder(s._computeOrder)
{
  for (int i=0; i<3; ++i) {
    _velocity[i]=s._velocity[i];
    _force[i]=s._force[i];
  }
  for (int i=0; i<N_SEG_DATA; ++i) {
    _segmentData._data[i]=s._segmentData._data[i];
  }
}

void Segment::loadBinary(FILE* inputDataFile, Branch* branch, const int segmentIndex)
{
  _branch = branch;
 
  _segmentIndex = segmentIndex;
 
  size_t s=fread(&_segmentData._sphere._coords, sizeof(double), 3, inputDataFile);
  for (int i=0; i<3; ++i) _segmentData._data[SEG_ORIG_COORDS+i]=_segmentData._sphere._coords[i];

  s=fread(&_segmentData._sphere._radius, sizeof(double), 1, inputDataFile);

  setKey();

  _mass = MIN_MASS;//(_segmentData._sphere._radius+MIN_MASS) * (_segmentData._sphere._radius+MIN_MASS) * (_segmentData._sphere._radius+MIN_MASS);
}

void Segment::loadText(double x, double y, double z, double r, Branch* branch, const int segmentIndex)
{
  _branch = branch;
  _segmentIndex = segmentIndex;
  _segmentData._sphere._coords[0]=x;
  _segmentData._sphere._coords[1]=y;
  _segmentData._sphere._coords[2]=z;
  _segmentData._sphere._radius=r;
  for (int i=0; i<3; ++i) _segmentData._data[SEG_ORIG_COORDS+i]=_segmentData._sphere._coords[i];

  setKey();
  _mass = MIN_MASS;//(_segmentData._sphere._radius+MIN_MASS) * (_segmentData._sphere._radius+MIN_MASS) * (_segmentData._sphere._radius+MIN_MASS);
}

void Segment::writeCoordinates(FILE* fp)
{
  fwrite(&_segmentData._sphere._coords, sizeof(double), 3, fp);
  fwrite(&_segmentData._sphere._radius, sizeof(double), 1, fp);
}

Segment& Segment::operator=(const Segment& s)
{
  if (this==&s) return *this;
  _mass=s._mass;
  _segmentIndex=s._segmentIndex;
  _segmentArrayIndex=s._segmentArrayIndex;
  _branch=s._branch;
  _isJunctionSegment=s._isJunctionSegment;
  _frontLevel=s._frontLevel;
  _computeOrder=s._computeOrder;

  for (int i=0; i<3; ++i) {
    _velocity[i]=s._velocity[i];
    _force[i]=s._force[i];
  }
  for (int i=0; i<N_SEG_DATA; ++i) {
    _segmentData._data[i]=s._segmentData._data[i];
  }
  return *this;
}

void Segment::resetBranch(Branch* b)
{
  _branch=b;
  setKey();
}

void Segment::setKey()
{
  _segmentData._sphere._key=_segmentDescriptor.getSegmentKey(this);
}

void Segment::addTouch(Touch* touch) 
{
  _touches.push_back(touch);
}

void Segment::resetTouches() 
{
  _touches.clear();
}

bool Segment::isTerminalSegment()
{
  return ( _branch->isTerminalBranch() && 
	   _segmentIndex==_branch->getNumberOfSegments()-1 );
}

void Segment::clearForce()
{
  _force[0]=_force[1]=_force[2] = 0.0;
}

void Segment::limitPrecision()
{
  double x=_segmentData._sphere._coords[0];
  double y=_segmentData._sphere._coords[1];
  double z=_segmentData._sphere._coords[2];
  double r=_segmentData._sphere._radius;

  _segmentData._sphere._coords[0]=floor(x*1000000.0)/1000000.0;
  _segmentData._sphere._coords[1]=floor(y*1000000.0)/1000000.0;
  _segmentData._sphere._coords[2]=floor(z*1000000.0)/1000000.0;
  _segmentData._sphere._radius=floor(r*1000000.0)/1000000.0;
}
