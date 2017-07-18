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
#include "KCaChannel.h"
#include "CG_KCaChannel.h"
#include "rndm.h"

#define SMALL 1.0E-6

#define SCHWEIGHOFER_1999

#ifdef  SCHWEIGHOFER_1999
// Pretty sure 210^-5 in the paper was supposed to be 2x10^-5
#define FAC_ALPHA 2.0E-5
#define MAX_ALPHA 0.01
#define BETA 0.015
#endif

void KCaChannel::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    float ca = (*Ca)[i];
#ifdef  SCHWEIGHOFER_1999
    //assert(FAC_ALPHA*ca<MAX_ALPHA);
    float as = FAC_ALPHA*ca;
    float ps = 0.5*dt*(as + BETA);
    s[i] = (dt*as + s[i]*(1.0 - ps))/(1.0 + ps);
    g[i] = gbar[i]*s[i];
#endif
  }
}

void KCaChannel::initializeKCaChannels(RNG& rng)
{
  unsigned size=branchData->size;
  assert(Ca);
  assert(gbar.size()==size);
  assert (Ca->size()==size);
  if (s.size()!=size) s.increaseSizeTo(size);
  if (g.size()!=size) g.increaseSizeTo(size);
  for (int i=0; i<size; ++i) {
    gbar[i]=gbar[0];
  }
  for (unsigned i=0; i<size; ++i) {
    float ca = (*Ca)[i];
#ifdef  SCHWEIGHOFER_1999
    assert(FAC_ALPHA*ca<MAX_ALPHA);
    float as = FAC_ALPHA*ca;
    s[i] = as/(as + BETA);
    g[i] = gbar[i]*s[i];
#endif
  }
}

KCaChannel::~KCaChannel()
{
}

