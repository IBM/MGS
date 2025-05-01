// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "ChannelHayMK.h"
#include "CG_ChannelHayMK.h"
#include "NumberUtils.h"
#include "rndm.h"

#define SMALL 1.0E-6

// Muscarinic-activated K+ outward current
//
#define AMC 0.0033
#define AMV 35.0
#define AMD 0.1
#define BMC 0.0033
#define BMV 35.0
#define BMD -0.1
#define T_ADJ 2.9529 // 2.3^((34-21)/10)


void ChannelHayMK::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i<branchData->size; ++i) {
    float v = (*V)[i];
    float am = AMC*exp(AMD*(v + AMV));
    float bm = BMC*exp(BMD*(v + BMV));
    float pm = 0.5*dt*(am + bm)*T_ADJ; 
    m[i] = (dt*am*T_ADJ + m[i]*(1.0 - pm))/(1.0 + pm);
    g[i] = gbar[i]*m[i];
  }
}

void ChannelHayMK::initialize(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert(V->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (m.size()!=size) m.increaseSizeTo(size);
  for (int i = 0; i<size; ++i) {
    gbar[i] = gbar[0];
  }
  for (unsigned i=0; i<size; ++i) {
    float v = (*V)[i];
    float am = AMC*exp(AMD*(v + AMV));
    float bm = BMC*exp(BMD*(v + BMV));
    m[i] = am/(am + bm);
    g[i] = gbar[i]*m[i];
  }
}

ChannelHayMK::~ChannelHayMK()
{
}

