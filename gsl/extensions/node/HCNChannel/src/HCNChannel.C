// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "HCNChannel.h"
#include "CG_HCNChannel.h"
#include "rndm.h"

#define SMALL 1.0E-6

//#define SCHWEIGHOFER_1999
#define DEGRUIJL_2012

#ifdef SCHWEIGHOFER_1999
#define QRN 75.0
#define QRD 5.5
#endif
#ifdef DEGRUIJL_2012
#define QRN 80.0
#define QRD 4.0
#endif


void HCNChannel::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    float v=(*V)[i];
    //assert((v + QRN)/QRD>0);
    float qinf = 1.0/(1 + exp((v + QRN)/QRD));
    float tauq = 1.0/(exp(-0.086*v - 14.6) + exp(0.07*v - 1.87));
    float pq = 0.5*dt/tauq;
    q[i] = (2.0*pq*qinf + q[i]*(1.0 - pq))/(1.0 + pq);
    g[i] = gbar[i]*q[i];
  }
}

void HCNChannel::initializeHCNChannels(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert(V->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (q.size()!=size) q.increaseSizeTo(size);
  for (int i=0; i<size; ++i) {
    gbar[i]=gbar[0];
  }
  for (unsigned i=0; i<size; ++i) {
    float v=(*V)[i];
    q[i] = 1.0/(1 + exp((v + QRN)/QRD));
    g[i] = gbar[i]*q[i];
  }
}

HCNChannel::~HCNChannel()
{
}

