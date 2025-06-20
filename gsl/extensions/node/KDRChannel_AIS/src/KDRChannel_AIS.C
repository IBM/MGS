// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "KDRChannel_AIS.h"
#include "CG_KDRChannel_AIS.h"
#include "ShallowArray.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"
#include "MaxComputeOrder.h"
#include "NumberUtils.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

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

