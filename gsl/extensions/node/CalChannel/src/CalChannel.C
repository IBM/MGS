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
#include "CalChannel.h"
#include "CG_CalChannel.h"
#include "rndm.h"
#include "NumberUtils.h"

#define SMALL 1.0E-6
//#define SCHWEIGHOFER_1999
#define DEGRUIJL_2012

#ifdef SCHWEIGHOFER_1999
#define TAUK 5.0
#endif
#ifdef DEGRUIJL_2012
#define TAUK 1.0
#endif




void CalChannel::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    E_Ca[i]=(0.04343 * *(getSharedMembers().T) * log(*(getSharedMembers().Ca_EC) / (*Ca_IC)[i]));
    float v=(*V)[i];
    float kinf = 1.0/(1.0 + exp(-(v + 61.0)/4.2));
    float tauk = TAUK;
    float linf = 1.0/(1.0 + exp((v + 85.5)/8.5));
    float taul = (20.0*(exp((v + 160.0)/30.0))/(1.0 + exp((v + 84.0)/7.3))) + 35.0;
    float pk = 0.5*dt/tauk;
    float pl = 0.5*dt/taul;
    k[i] = (2.0*pk*kinf + k[i]*(1.0 - pk))/(1.0 + pk);
    l[i] = (2.0*pl*linf + l[i]*(1.0 - pl))/(1.0 + pl);
    g[i] = gbar[i]*k[i]*k[i]*k[i]*l[i];
    I_Ca[i] = g[i] * (v-E_Ca[i]);
  }
}

void CalChannel::initializeCalChannels(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert(Ca_IC);
  assert(gbar.size()==size);
  assert(V->size()==size);
  assert(Ca_IC->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (k.size()!=size) k.increaseSizeTo(size);
  if (l.size()!=size) l.increaseSizeTo(size);
  if (E_Ca.size()!=size) E_Ca.increaseSizeTo(size);
  if (I_Ca.size()!=size) I_Ca.increaseSizeTo(size);
  for (int i=0; i<size; ++i) {
    gbar[i]=gbar[0];
  }
  for (unsigned i=0; i<size; ++i) {
    float v=(*V)[i];
    k[i] = 1.0/(1.0 + exp(-(v + 61.0)/4.2));
    l[i] = 1.0/(1.0 + exp( (v + 85.5)/8.5));
    g[i] = gbar[i]*k[i]*k[i]*k[i]*l[i];
  }
  assert (getSharedMembers().T!=0 && getSharedMembers().Ca_EC!=0);
}

CalChannel::~CalChannel()
{
}

