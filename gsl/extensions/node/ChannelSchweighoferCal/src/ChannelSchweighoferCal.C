// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "ChannelSchweighoferCal.h"
#include "CG_ChannelSchweighoferCal.h"
#include "rndm.h"

#define SMALL 1.0E-6

#define IMV 61.0
#define IMD 4.2
#define IHV 85.5
#define IHD 8.5

float ChannelSchweighoferCal::vtrap(float x, float y) {
  return(fabs(x/y) < SMALL ? y*(x/y/2 - 1) : x/(1 - exp(x/y)));
}

void ChannelSchweighoferCal::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    E_Ca[i]=(0.04343 * *(getSharedMembers().T) * log(*(getSharedMembers().Ca_EC) / (*Ca_IC)[i]));
    float v=(*V)[i];
    float minf = 1.0/(1.0 + exp(-(v + IMV)/IMD));
    float taum = 5.0;
    float hinf = 1.0/(1.0 + exp((v + IHV)/IHD));
    float tauh = (20.0*(exp((v + 160.0)/30.0))/(1.0 + exp((v + 84.0)/7.3))) + 35.0;
    float pm = 0.5*dt/taum;
    float ph = 0.5*dt/tauh;
    m[i] = (2.0*pm*minf + m[i]*(1.0 - pm))/(1.0 + pm);
    h[i] = (2.0*ph*hinf + h[i]*(1.0 - ph))/(1.0 + ph);
    g[i] = gbar[i]*m[i]*m[i]*m[i]*h[i];
    I_Ca[i] = g[i] * (v-E_Ca[i]);
  }
}

void ChannelSchweighoferCal::initialize(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert(Ca_IC);
  assert(gbar.size()==size);
  assert(V->size()==size);
  assert(Ca_IC->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (m.size()!=size) m.increaseSizeTo(size);
  if (h.size()!=size) h.increaseSizeTo(size);
  if (E_Ca.size()!=size) E_Ca.increaseSizeTo(size);
  if (I_Ca.size()!=size) I_Ca.increaseSizeTo(size);
  for (int i=0; i<size; ++i) {
    gbar[i]=gbar[0];
  }
  for (unsigned i=0; i<size; ++i) {
    float v=(*V)[i];
    m[i] = 1.0/(1.0 + exp(-(v + IMV)/IMD));
    h[i] = 1.0/(1.0 + exp( (v + IHV)/IHD));
    g[i] = gbar[i]*m[i]*m[i]*m[i]*h[i];
  }
  assert (getSharedMembers().T!=0 && getSharedMembers().Ca_EC!=0);
}

ChannelSchweighoferCal::~ChannelSchweighoferCal()
{
}

