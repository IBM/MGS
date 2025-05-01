// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "KDRChannel_IO.h"
#include "CG_KDRChannel_IO.h"
#include "rndm.h"

#define SMALL 1.0E-6

#define DEGRUIJL_2012


void KDRChannel_IO::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    float v=(*V)[i];
    float ninf = 1.0/(1.0 + exp(-(v + 3.0)/10.0));
    float pinf = 1.0/(1.0 + exp( (v + 51.0)/12.0));
    float taupn = (47.0*(exp(-(v + 50.0)/900.0) ) ) + 5.0;
    float pn = 0.5*dt/taupn;
    n[i] = (2.0*pn*ninf + n[i]*(1.0 - pn))/(1.0 + pn);
    p[i] = (2.0*pn*pinf + p[i]*(1.0 - pn))/(1.0 + pn);
    g[i] = gbar[i]*n[i]*p[i];
  }
}

void KDRChannel_IO::initializeKDRChannels(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert (V->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (n.size()!=size) n.increaseSizeTo(size);
  if (p.size()!=size) p.increaseSizeTo(size);
  for (unsigned i=0; i<size; ++i) {
    gbar[i]=gbar[0];
    float v=(*V)[i];
    n[i] = 1.0/(1.0 + exp(-(v + 3.0)/10.0));
    p[i] = 1.0/(1.0 + exp( (v + 51.0)/12.0));
    g[i]=gbar[i]*n[i]*p[i];
  }
}

KDRChannel_IO::~KDRChannel_IO()
{
}

