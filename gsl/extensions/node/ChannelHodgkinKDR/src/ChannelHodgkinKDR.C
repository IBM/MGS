// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "ChannelHodgkinKDR.h"
#include "CG_ChannelHodgkinKDR.h"
#include "rndm.h"
#include "NumberUtils.h"

#define SMALL 1.0E-6

#define AMC 0.01
#define AMV 55.0
#define AMD 10.0
#define BMC 0.125
#define BMV 65.0
#define BMD 80.0


void ChannelHodgkinKDR::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    float v=(*V)[i];
    float am = AMC*vtrap(-(v + AMV), AMD);
    float bm = BMC*exp(-(v + BMV)/BMD);
    float pm = 0.5*dt*(am + bm);
    m[i] = (dt*am + m[i]*(1.0 - pm))/(1.0 + pm);
    g[i] = gbar[i]*m[i]*m[i]*m[i]*m[i];
  }
}

void ChannelHodgkinKDR::initialize(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert (V->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (m.size()!=size) m.increaseSizeTo(size);
  for (int i=0; i<size; ++i) {
    gbar[i]=gbar[0];
  }
  for (unsigned i=0; i<size; ++i) {
    float v=(*V)[i];
    float am = AMC*vtrap(-(v + AMV), AMD);
    float bm = BMC*exp(-(v + BMV)/BMD);
    m[i] = am/(am + bm);
    g[i]=gbar[i]*m[i]*m[i]*m[i]*m[i];
  }
}

ChannelHodgkinKDR::~ChannelHodgkinKDR()
{
}

