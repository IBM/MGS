// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
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

