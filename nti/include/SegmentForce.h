// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef SEGMENTFORCE_H
#define SEGMENTFORCE_H

#include "SegmentDescriptor.h"
#include "Sphere.h"

#include <mpi.h>
#include <vector>

#define N_SEGFORCE_DATA 4

class SegmentForce
{
 public:
  SegmentForce();

  void
  HarmonicDistance(double * siteA, double *siteB, 
	     double R0, double k0, 
	     double & E,double * ForceOnA );

  void          //  origin, end1, end2
  HarmonicAngle(double * siteA, double *siteB, double *siteC,
	  double R0, double k0, double & E,
	  double * ForceOnA, double * ForceOnB, double * ForceOnC );


  void          
  LennardJonesForce(Sphere const & sphereA, Sphere const & sphereB,
	      double sigma, double epsilon, double & E,
	      double * ForceOnA );
    
  void          
  RepulsiveForce(Sphere const & sphereA, Sphere const & sphereB,
	   double sigma, double epsilon, double & E,
	   double * ForceOnA );
	   
  void          
  SignalInteractionForce(Sphere const & sphereA, Sphere const & sphereB,
	   double sigma, double epsilon, double & E,
	   double * ForceOnA );
	   
  ~SegmentForce();
		
  
  //TUAN: NOTE - potential bug when the key's size change
  //  PLAN: move key component to the last part of the array
  key_size_t getKey() {return _segmentForceData[0];}
  const double getForceX() {return _segmentForceData[1];}
  const double getForceY() {return _segmentForceData[2];}
  const double getForceZ() {return _segmentForceData[3];}
  double* getForces() {return &_segmentForceData[1];}

  double* getSegmentForceData() {return _segmentForceData;}

  void setKey(key_size_t key) {_segmentForceData[0]=key;}
  void setForceX(double forceX) {_segmentForceData[1]=forceX;}
  void setForceY(double forceY) {_segmentForceData[2]=forceY;}
  void setForceZ(double forceZ) {_segmentForceData[3]=forceZ;}

  void printSegmentForce();
	  
 private:
  double _segmentForceData[N_SEGFORCE_DATA];
  static SegmentDescriptor _segmentDescriptor;
};

#endif
