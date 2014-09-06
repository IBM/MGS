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
// =================================================================

#include "Lens.h"
#include "ChannelHaySK.h"
#include "CG_ChannelHaySK.h"
#include "rndm.h"
#include <math.h>

#define SMALL 1.0E-6

#define FACTOR 0.43
#define POWER 4.8
#define TAU 1.0

void ChannelHaySK::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    float ca = (*Ca)[i];
    float minf = (1/TAU)/(1 + pow(FACTOR/ca,POWER));
    float pm = 0.5*dt/TAU;
    m[i] = (dt*minf + m[i]*(1.0 - pm))/(1.0 + pm);
    g[i] = gbar[i]*m[i];    
  }
}

void ChannelHaySK::initialize(RNG& rng)
{
  unsigned size=branchData->size;
  assert(Ca);
  assert(gbar.size()==size);
  assert(Ca->size()==size);
  if (m.size()!=size) m.increaseSizeTo(size);
  if (g.size()!=size) g.increaseSizeTo(size);
  for (int i=0; i<size; ++i) {
    gbar[i]=gbar[0];
  }
  for (unsigned i=0; i<size; ++i) {
    float ca = (*Ca)[i];
    m[i] = 1/(1 + pow(FACTOR/ca,POWER));
    g[i] = gbar[i]*m[i];
  }
}

ChannelHaySK::~ChannelHaySK()
{
}

