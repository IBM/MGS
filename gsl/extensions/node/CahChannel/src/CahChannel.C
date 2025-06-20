// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "CahChannel.h"
#include "CG_CahChannel.h"
#include "rndm.h"
#include "NumberUtils.h"

#define SMALL 1.0E-6
//#define SCHWEIGHOFER_1999
#define DEGRUIJL_2012

#ifdef SCHWEIGHOFER_1999
#define ARC 1.6
#define ARV -5.0
#define ARD 14.0
#define BRC 0.02
#define BRV 8.5
#define BRD 5.0
#endif
#ifdef DEGRUIJL_2012
#define ARC 0.34
#define ARV -5.0
#define ARD 13.9
#define BRC 0.004
#define BRV 8.5
#define BRD 5.0
#endif


void CahChannel::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    E_Ca[i]=(0.04343 * *(getSharedMembers().T) * log(*(getSharedMembers().Ca_EC) / (*Ca_IC)[i]));
    float v=(*V)[i];
    float ar = ARC/(1.0 + exp(-(v + ARV)/ARD));
    float br = BRC*vtrap(v + BRV, BRD);
    float pr = 0.5*dt*(ar + br);
    r[i] = (dt*ar + r[i]*(1.0 - pr))/(1.0 + pr);
    g[i]=gbar[i]*r[i]*r[i];
    I_Ca[i] = g[i] * (v-E_Ca[i]);
  }
}

void CahChannel::initialize(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert(V->size()==size);
  assert(Ca_IC->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (r.size()!=size) r.increaseSizeTo(size);
  if (E_Ca.size()!=size) E_Ca.increaseSizeTo(size);
  if (I_Ca.size()!=size) I_Ca.increaseSizeTo(size);
  for (int i=0; i<size; ++i) {
    gbar[i]=gbar[0];
  }
  for (unsigned i=0; i<size; ++i) {
    float v=(*V)[i];
    float ar = ARC/(1.0 + exp(-(v + ARV)/ARD));
    float br = BRC*vtrap(v + BRV, BRD);
    r[i] = ar/(ar + br);
    g[i] = gbar[i]*r[i]*r[i];
  }
  assert (getSharedMembers().T!=0 && getSharedMembers().Ca_EC!=0);
}

CahChannel::~CahChannel()
{
}

