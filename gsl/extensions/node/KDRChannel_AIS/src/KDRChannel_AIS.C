// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "KDRChannel_AIS.h"
#include "CG_KDRChannel_AIS.h"
#include "ShallowArray.h"
#include "rndm.h"

#define SMALL 1.0E-6

#define DEGRUIJL_2012

/* Degruijl 2012 used the following parameters:
   gbar_0 = 5.0;
   gbar_AIS = 20.0;
*/

// Fast Component AND Axon hillock
#define ANC 0.13
#define ANV 25.0
#define AND 10.0
#define BNC 1.69
#define BNV 35.0
#define BND 80.0


float KDRChannel_AIS::vtrap(float x, float y) {
	return(fabs(x/y) < SMALL ? y*(1 - x/y/2) : x/(exp(x/y) - 1));
}

void KDRChannel_AIS::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    float v=(*V)[i];
    float an = ANC*vtrap(-(v + ANV), AND);
    float bn = BNC*exp(-(v + BNV)/BND);
    float pn = 0.5*dt*(an + bn);
    n[i] = (dt*an + n[i]*(1.0 - pn))/(1.0 + pn);
    g[i] = gbar[i]*n[i]*n[i]*n[i]*n[i];
  }
}

void KDRChannel_AIS::initializeKDRChannels(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert (V->size()==size);
  if (gbar.size()!=size) gbar.increaseSizeTo(size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (n.size()!=size) n.increaseSizeTo(size);
  ShallowArray<DimensionStruct*>& dimensionsArray = *dimensions;
  for (unsigned i=0; i<size; ++i) {
    gbar[i]=getSharedMembers().gbar_0;
    if (dimensionsArray[i]->dist2soma<=getSharedMembers().d_AIS) {
      gbar[i]=getSharedMembers().gbar_AIS;
    }
    gbar[i]=gbar[0];
    float v=(*V)[i];
    float an = ANC*vtrap(-(v + ANV), AND);
    float bn = BNC*exp(-(v + BNV)/BND);
    n[i] = an/(an + bn);
    g[i]=gbar[i]*n[i]*n[i]*n[i]*n[i];
  }
}

KDRChannel_AIS::~KDRChannel_AIS()
{
}

