// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "ChannelSchweighoferKCa.h"
#include "CG_ChannelSchweighoferKCa.h"
#include "rndm.h"
#include <math.h>

#define SMALL 1.0E-6

// Pretty sure 210^-5 in the paper was supposed to be 2x10^-5
#define FAC_ALPHA 2.0E-5
#define MAX_ALPHA 0.01
#define BETA 0.015

void ChannelSchweighoferKCa::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    float ca = (*Ca)[i];
    assert(FAC_ALPHA*ca<MAX_ALPHA);
    float am = FAC_ALPHA*ca;
    float pm = 0.5*dt*(am + BETA);
    m[i] = (dt*am + m[i]*(1.0 - pm))/(1.0 + pm);
    g[i] = gbar[i]*m[i];
  }
}

void ChannelSchweighoferKCa::initialize(RNG& rng)
{
  unsigned size=branchData->size;
  assert(Ca);
  assert(gbar.size()==size);
  assert (Ca->size()==size);
  if (m.size()!=size) m.increaseSizeTo(size);
  if (g.size()!=size) g.increaseSizeTo(size);
  for (int i=0; i<size; ++i) {
    gbar[i]=gbar[0];
  }
  for (unsigned i=0; i<size; ++i) {
    float ca = (*Ca)[i];
    assert(FAC_ALPHA*ca<MAX_ALPHA);
    float am = FAC_ALPHA*ca;
    m[i] = am/(am + BETA);
    g[i] = gbar[i]*m[i];
  }
}

ChannelSchweighoferKCa::~ChannelSchweighoferKCa()
{
}
