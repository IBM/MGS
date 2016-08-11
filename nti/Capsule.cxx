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

#include "Capsule.h"
#include "Segment.h"
#include "VecPrim.h"
#include "math.h"

SegmentDescriptor Capsule::_segmentDescriptor;

Capsule::Capsule()
  : _branch(0)
{
  for (int i=0; i<N_CAP_DATA; ++i) {
    _capsuleData._data[i]=0;
  }
}

Capsule::~Capsule()
{
}


Capsule::Capsule(Capsule const & c)
  : _branch(c._branch)
{
  memcpy(_capsuleData._data, c._capsuleData._data, N_CAP_DATA*sizeof(double));
}

bool Capsule::operator<(const Capsule& c1) const
{
  bool rval=false;
  double key0=getKey();
  double key1=c1.getKey();

  unsigned int n0=_segmentDescriptor.getNeuronIndex(key0);
  unsigned int n1=_segmentDescriptor.getNeuronIndex(key1);

  if (n0==n1) {
    unsigned int b0=_segmentDescriptor.getBranchIndex(key0);
    unsigned int b1=_segmentDescriptor.getBranchIndex(key1);
 
    if (b0==b1) {
      unsigned int s0=_segmentDescriptor.getSegmentIndex(key0);
      unsigned int s1=_segmentDescriptor.getSegmentIndex(key1);

      rval=(s0<s1);
      
    }
    else rval=(b0<b1);

  }
  else rval=(n0<n1);

  return rval;
}

double Capsule::getEndProp()
{
  double dist=SqDist(getBeginCoordinates(), getEndCoordinates());
  double rval=(dist>0) ? 1.0-(getRadius()/dist) : 0;
  return (rval<0) ? 0 : rval;
}

void Capsule::getEndSphere(Sphere& sphere)
{
  sphere=_capsuleData._sphere;
  double* endCoords=getEndCoordinates();
  sphere._coords[0]=endCoords[0];
  sphere._coords[1]=endCoords[1];
  sphere._coords[2]=endCoords[2];
}

int Capsule::getEndSphereSurfaceArea()
{
  double r=getRadius();
  return (4.0*M_PI*r*r);
}

void Capsule::readFromFile(FILE* dataFile)
{
  size_t s=fread(_capsuleData._data, sizeof(double), N_CAP_DATA, dataFile);
}

void Capsule::writeToFile(FILE* dataFile)
{
  fwrite(_capsuleData._data, sizeof(double), N_CAP_DATA, dataFile);
}

bool Capsule::operator==(const Capsule& c1) const
{
  double key0=getKey();
  double key1=c1.getKey();

  unsigned int n0=_segmentDescriptor.getNeuronIndex(key0);
  unsigned int n1=_segmentDescriptor.getNeuronIndex(key1);

  unsigned int b0=_segmentDescriptor.getBranchIndex(key0);
  unsigned int b1=_segmentDescriptor.getBranchIndex(key1);

  unsigned int i0=_segmentDescriptor.getSegmentIndex(key0);
  unsigned int i1=_segmentDescriptor.getSegmentIndex(key1);

  return (n0==n1 && b0==b1 && i0==i1);
}

int Capsule::getSurfaceArea()
{
  double dist=sqrt(SqDist(getBeginCoordinates(), getEndCoordinates()));
  return (dist*2.0*M_PI*getRadius());
}
