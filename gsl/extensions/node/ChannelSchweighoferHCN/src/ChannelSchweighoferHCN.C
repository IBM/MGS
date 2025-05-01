// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "ChannelSchweighoferHCN.h"
#include "CG_ChannelSchweighoferHCN.h"
#include "rndm.h"
#include "NumberUtils.h"

#define SMALL 1.0E-6


void ChannelSchweighoferHCN::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    float v=(*V)[i];
    assert((v + 75.0)/5.5>0);
    float minf = 1.0/(1 + exp((v + 75.0)/5.5));
    float taum = 1.0/(exp(-0.086*v - 14.6) + exp(0.07*v - 1.87));
    float pm = 0.5*dt/taum;
    m[i] = (2.0*pm*minf + m[i]*(1.0 - pm))/(1.0 + pm);
    g[i] = gbar[i]*m[i];
  }
}

void ChannelSchweighoferHCN::initialize(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert(V->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (m.size()!=size) m.increaseSizeTo(size);
  for (int i=0; i<size; ++i) {
    gbar[i]=gbar[0];
  }
  for (unsigned i=0; i<size; ++i) {
    float v=(*V)[i];
    m[i] = 1.0/(1 + exp((v + 75.0)/5.5));
    g[i] = gbar[i]*m[i];
  }
}

ChannelSchweighoferHCN::~ChannelSchweighoferHCN()
{
}

