// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "SegmentForce.h"
#include <limits.h>
#include "VecPrim.h"

#define BondEps 1e-7
#define MAX_FORCE  1000.0

#define MIN_FAC 0.001

SegmentDescriptor SegmentForce::_segmentDescriptor;

#include <cassert>
#include <iostream>

SegmentForce::SegmentForce()
{
  for (int i=0; i<N_SEGFORCE_DATA; ++i) {
    _segmentForceData[i]=0;
  }
}

void SegmentForce::HarmonicDistance(double *siteA, 
				    double *siteB, 
				    double R0, 
				    double k0, 
				    double & E, 
				    double * ForceOnA )
{
  double Dist  = SqDist(siteA,siteB);
  Dist = sqrt(Dist);
  double DeltaFromEquilibrium = Dist - R0;
 
  E = k0 * DeltaFromEquilibrium * DeltaFromEquilibrium;
  
  double DEDR  =  (Dist > BondEps) ?  -2.0 * ( k0 * DeltaFromEquilibrium ) / Dist : 0.0;
  
  for(int i = 0; i < 3; i++) {
    ForceOnA[i] = DEDR * ( siteB[i] - siteA[i] );
  }
}

//  origin, end1, end2
void SegmentForce::HarmonicAngle(double *siteA, 
				 double *siteB, 
				 double *siteC,
				 double R0, 
				 double k0, 
				 double & E,
				 double * ForceOnA, 
				 double * ForceOnB, 
				 double * ForceOnC )
{
  /////DETERMINE COS(THETA) = R12 DOT R32 / SQRT((R12**2)*(R32**2))
  
  double DistanceAB  = SqDist(siteA,siteB); DistanceAB = sqrt(DistanceAB);
  double DistanceAC  = SqDist(siteA,siteC); DistanceAC = sqrt(DistanceAC);
  
  // These vectors will have been computed in the MDVM as a
  // byproduct of the above distances.
  double VectorAB[3], VectorAC[3];
  
  for(int i = 0; i < 3; i++) {
    VectorAB[i] = siteB[i] - siteA[i];
    VectorAC[i] = siteC[i] - siteA[i];
  }
  
  double dotABAC = Vec3Dot(VectorAB, VectorAC);
  double COSTHE = dotABAC / (DistanceAB * DistanceAC);
  
  /////   DETERMINE THETA.  IT IS IN THE RANGE OF 0 TO PI RADIANS.
  double THETA = acos(COSTHE);
  E = k0 * ( THETA - R0 ) * (THETA - R0);
  double DEDTH = 2.0 * k0 * (THETA - R0);
    
  // COMPUTE AND NORMALIZE R(PERP) TO FACILITATE FORCE EVALUATION
  // R(PERP) = R32 CROSS R12
    
  double RP[3];
  RP[0] = VectorAC[1] * VectorAB[2] - VectorAC[2] * VectorAB[1];
  RP[1] = VectorAC[2] * VectorAB[0] - VectorAC[0] * VectorAB[2];
  RP[2] = VectorAC[0] * VectorAB[1] - VectorAC[1] * VectorAB[0];
    
  // RPERP WILL NOT NORMALIZE IF 1-2-3 COLLINEAR; I.E., THETA=0 OR 180
  // FORCE CAN BE LEGITIMATELY SET TO ZERO IN THIS CASE.
	
  double RPL = 0.0;
  for(int i = 0; i < 3; i++) {
    RPL += (RP[i]*RP[i]);
  }
  RPL = sqrt(RPL);
  if( RPL > 0.0000000001 )
    RPL = 1.0 / RPL;
  else
    RPL=0.0;
    
  //    COMPUTE GRADIENTS OF ENERGY WITH RESPECT TO COMPONENTS OF VectorAB,RO2
  double    R12R = -1.0 /  Vec3Dot(VectorAB,VectorAB);
  double    R32R =  1.0 /  Vec3Dot(VectorAC,VectorAC);
  double    DTD12X = R12R*(RPL*DEDTH) * (VectorAB[1]*RP[2] - VectorAB[2]*RP[1]);
  double    DTD12Y = R12R*(RPL*DEDTH) * (VectorAB[2]*RP[0] - VectorAB[0]*RP[2]);
  double    DTD12Z = R12R*(RPL*DEDTH) * (VectorAB[0]*RP[1] - VectorAB[1]*RP[0]);
  double    DTD32X = R32R*(RPL*DEDTH) * (VectorAC[1]*RP[2] - VectorAC[2]*RP[1]);
  double    DTD32Y = R32R*(RPL*DEDTH) * (VectorAC[2]*RP[0] - VectorAC[0]*RP[2]);
  double    DTD32Z = R32R*(RPL*DEDTH) * (VectorAC[0]*RP[1] - VectorAC[1]*RP[0]);
  
  // COMPUTE FORCES DUE TO ANGLE ENERGY ON O, H1 AND H2
  ForceOnB[0] = DTD12X;
  ForceOnB[1] = DTD12Y;
  ForceOnB[2] = DTD12Z;
  ForceOnA[0] = -DTD12X-DTD32X;
  ForceOnA[1] = -DTD12Y-DTD32Y;
  ForceOnA[2] = -DTD12Z-DTD32Z;
  ForceOnC[0] = DTD32X;
  ForceOnC[1] = DTD32Y;
  ForceOnC[2] = DTD32Z;
  return;
}

//  origin, end1, end2
void SegmentForce::LennardJonesForce(Sphere const & sphereA,
				     Sphere const & sphereB, 
				     double sigma,
				     double epsilon, 
				     double & E,
				     double *ForceOnA )
{
  double dVecABMag2 = 0;
  double dVecAB[3];
  for(int i = 0; i < 3; i++) {
    dVecAB[i] = sphereB._coords[i] - sphereA._coords[i];
    dVecABMag2 += dVecAB[i]*dVecAB[i];
  }

  double facABMag = 1-((sphereB._radius + sphereA._radius)/sqrt(dVecABMag2));
  if (facABMag<MIN_FAC) facABMag=MIN_FAC;
  dVecABMag2 = 0;

  for(int i = 0; i < 3; i++) {
    dVecAB[i] *= facABMag;
    dVecABMag2 += dVecAB[i]*dVecAB[i];
  }

  double dVecABMagR = 1.0/sqrt(dVecABMag2);
  double dVecABMag2R = dVecABMagR*dVecABMagR;

  //  double dVecABMag = dVecABMag2 * dVecABMagR;
  
  double tmp2          = sigma*sigma*dVecABMag2R;
  double tmp6          = tmp2*tmp2*tmp2;
  double tmp12         = tmp6 * tmp6;
  
  double pdSdR ;
  //  SwitchFunctionRadii sfr;
  //  aMDVM.GetSwitchFunctionRadii(sfr);
  
  //  SwitchFunction sf(sfr) ;
  //  sf.Evaluate(dVecABMag,S,pdSdR) ;
  
  E = epsilon * (tmp12 - 2.0 * tmp6);
  
  double DEDR =
    (
     (12 * epsilon) *
     (
      (  tmp12)
      - (  tmp6)
      )
     ) * dVecABMagR ;
  
  for(int i = 0; i < 3 ; i++) {
    ForceOnA[0] = DEDR * dVecAB[i];
  }
  return;
}

//  origin, end1, end2
void SegmentForce::RepulsiveForce(Sphere const & sphereA,
				  Sphere const & sphereB, 
				  double sigma, 
				  double epsilon, 
				  double & E, 
				  double * ForceOnA )
{
  double dVecABMag2 = 0;
  double dVecAB[3];
  for(int i = 0; i < 3; i++) {
    dVecAB[i] = sphereB._coords[i] - sphereA._coords[i];
    dVecABMag2 += dVecAB[i]*dVecAB[i];
  }

  double facABMag = 1-((sphereB._radius + sphereA._radius)/sqrt(dVecABMag2));
  if (facABMag<MIN_FAC) facABMag=MIN_FAC;
  dVecABMag2 = 0;
			 
  for(int i = 0; i < 3; i++) {
    dVecAB[i] *= facABMag;
    dVecABMag2 += dVecAB[i]*dVecAB[i];
  }

  double dVecABMagR = 1.0/sqrt(dVecABMag2);
  double dVecABMag2R = dVecABMagR*dVecABMagR;
  // double dVecABMag = dVecABMag2 * dVecABMagR;
  
  double tmp2 = sigma*sigma*dVecABMag2R;
  double tmp6 = tmp2*tmp2*tmp2;
  // double tmp12 = tmp6 * tmp6;

  double pdSdR;
  //  SwitchFunctionRadii sfr;
  //  aMDVM.GetSwitchFunctionRadii(sfr);
  
  //  SwitchFunction sf(sfr) ;
  //  sf.Evaluate(dVecABMag,S,pdSdR) ;
  //  double lje = epsilon * tmp12 ;
  E = epsilon * tmp6;
  
  //  double DEDR = 12 * epsilon * tmp12 * dVecABMagR ;
  double DEDR = -6 * epsilon * tmp6 * dVecABMagR ;

  if(DEDR < -MAX_FORCE) {
    //printf("DEDR = %f set to %f\n", DEDR,-MAX_FORCE);
    DEDR = -MAX_FORCE;
  }
  else if(DEDR > MAX_FORCE) {
    //printf("DEDR = %f set to %f\n", DEDR, MAX_FORCE);
    DEDR = MAX_FORCE;
  }
  
  for(int i = 0; i < 3 ; i++) {
    ForceOnA[i] = DEDR * dVecAB[i];
  }
  return;
}

//  origin, end1, end2
void SegmentForce::SignalInteractionForce(Sphere const & sphereA,
				  Sphere const & sphereB, 
                                  double sigma,
                                  double epsilon,
                                  double & E,
                                  double * ForceOnA )
{
  double dVecABMag2 = 0;
  double dVecAB[3];
  for(int i = 0; i < 3; i++) {
    dVecAB[i] = sphereB._coords[i] - sphereA._coords[i];
    dVecABMag2 += dVecAB[i]*dVecAB[i];
  }

  double facABMag = 1-((sphereB._radius + sphereA._radius)/sqrt(dVecABMag2));
  if (facABMag<MIN_FAC) facABMag=MIN_FAC;
  dVecABMag2 = 0;
			 
  for(int i = 0; i < 3; i++) {
    dVecAB[i] *= facABMag;
    dVecABMag2 += dVecAB[i]*dVecAB[i];
  }

  double dVecABMagR = 1.0/sqrt(dVecABMag2);
  double dVecABMag2R = dVecABMagR*dVecABMagR;
  //  double dVecABMag = dVecABMag2 * dVecABMagR;

  double tmp2 = sigma*sigma*dVecABMag2R;
  //  double tmp6 = tmp2*tmp2*tmp2;
  //  double tmp12 = tmp6 * tmp6;

  double pdSdR;
  //  SwitchFunctionRadii sfr;
  //  aMDVM.GetSwitchFunctionRadii(sfr);

  //  SwitchFunction sf(sfr) ;
  //  sf.Evaluate(dVecABMag,S,pdSdR) ;
  //  double lje           = epsilon * tmp12 ;
  //  double lje = epsilon * tmp6;
  E = epsilon * tmp2;

  //  double DEDR = 12 * epsilon * tmp12 * dVecABMagR ;
  //  double DEDR = -6 * epsilon * tmp6 * dVecABMagR ;
  double DEDR = -2 * epsilon * tmp2 * dVecABMagR ;
  
  if(DEDR < -MAX_FORCE) {
    //printf("DEDR = %f set to %f\n", DEDR,-MAX_FORCE);
    DEDR = -MAX_FORCE;
  }
  else if(DEDR > MAX_FORCE) {
    //printf("DEDR = %f set to %f\n", DEDR, MAX_FORCE);
    DEDR = MAX_FORCE;
  
}
  for(int i = 0; i < 3 ; i++) {
    ForceOnA[i] = DEDR * dVecAB[i];
  }
  return;
}

void SegmentForce::printSegmentForce() 
{
  std::cerr<<_segmentDescriptor.getNeuronIndex(_segmentForceData[0])<<" "
           <<_segmentDescriptor.getBranchIndex(_segmentForceData[0])<<" "
           <<_segmentDescriptor.getSegmentIndex(_segmentForceData[0])<<" "
           <<_segmentForceData[1]<<" "
           <<_segmentForceData[2]<<" "
           <<_segmentForceData[3]<<std::endl;
}

SegmentForce::~SegmentForce()
{
}
