// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "KDRChannel.h"
#include "CG_KDRChannel.h"
#include "rndm.h"

#define SMALL 1.0E-6

#define HH_1952
//#define SCHWEIGHOFER_1999

#ifdef HH_1952
#define ANC 0.01
#define ANV 55.0
#define AND 10.0
#define BNC 0.125
#define BNV 65.0
#define BND 80.0
#endif
#ifdef SCHWEIGHOFER_1999
#define ANC 0.2
#define ANV 41.0
#define AND 10.0
#define BNC 2.5
#define BNV 51.0
#define BND 80.0
#endif


void KDRChannel::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    float v=(*V)[i];
    float an = ANC*vtrap(-(v + ANV), AND);
    float bn = BNC*exp(-(v + BNV)/BND);
    float pn = 0.5*dt*(an + bn);
    n[i] = (dt*an + n[i]*(1.0 - pn))/(1.0 + pn);
    g[i] = gbar[i]*n[i]*n[i]*n[i]*n[i];
  }
}

void KDRChannel::initializeKDRChannels(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert (V->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (n.size()!=size) n.increaseSizeTo(size);
  for (unsigned i=0; i<size; ++i) {
    gbar[i]=gbar[0];
    float v=(*V)[i];
    float an = ANC*vtrap(-(v + ANV), AND);
    float bn = BNC*exp(-(v + BNV)/BND);
    n[i] = an/(an + bn);
    g[i]=gbar[i]*n[i]*n[i]*n[i]*n[i];
  }
}

KDRChannel::~KDRChannel()
{
}

