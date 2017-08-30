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
#include "ChannelHayKv31.h"
#include "CG_ChannelHayKv31.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "NumberUtils.h"
#define SMALL 1.0E-6

#define IMV -18.7
#define IMD 9.7
#define TMF 4.0
#define TMV 46.56
#define TMD 44.14
#define T_ADJ 2.9529 // 2.3^((34-21)/10)


void ChannelHayKv31::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i<branchData->size; ++i) {
    float v = (*V)[i];
    float minf = 1.0/(1.0 + exp(-(v + IMV)/IMD));
    // float taum = TMF/(T_ADJ*(1 + exp(-(v + TMV)/TMD)));
    float taum = TMF/(1 + exp(-(v + TMV)/TMD));
    float pm = 0.5*dt/taum;
    m[i] = (2.0*pm*minf + m[i]*(1.0 - pm))/(1.0 + pm);
    if (m[i] < 0.0) {
      m[i] = 0.0;
    } else if (m[i] > 1.0) {
      m[i] = 1.0;
    }
    g[i] = gbar[i]*m[i];
#ifdef WAIT_FOR_REST
		float currentTime = getSimulation().getIteration() * (*getSharedMembers().deltaT);
		if (currentTime < NOGATING_TIME)
			g[i]= 0.0;
#endif
  }
}

void ChannelHayKv31::initialize(RNG& rng)
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
    m[i] = 1.0/(1.0 + exp(-(v + IMV)/IMD));
    g[i] = gbar[i]*m[i];
  }
}

ChannelHayKv31::~ChannelHayKv31()
{
}

