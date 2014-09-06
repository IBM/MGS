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
#include "ChannelSchweighoferCah.h"
#include "CG_ChannelSchweighoferCah.h"
#include "rndm.h"

#define SMALL 1.0E-6

#define AMC 1.6
#define AMV -5.0
#define AMD 14.0
#define BMC 0.02
#define BMV 8.5
#define BMD 5.0

float ChannelSchweighoferCah::vtrap(float x, float y) {
  return(fabs(x/y) < SMALL ? y*(1 - x/y/2) : x/(exp(x/y) - 1));
}

void ChannelSchweighoferCah::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    E_Ca[i]=(0.04343 * *(getSharedMembers().T) * log(*(getSharedMembers().Ca_EC) / (*Ca_IC)[i]));
    float v=(*V)[i];
    float am = AMC/(1.0 + exp(-(v + AMV)/AMD));
    float bm = BMC*vtrap(v + BMV, BMD);
    float pm = 0.5*dt*(am + bm);
    m[i] = (dt*am + m[i]*(1.0 - pm))/(1.0 + pm);
    g[i]=gbar[i]*m[i]*m[i];
    I_Ca[i] = g[i] * (v-E_Ca[i]);
  }
}

void ChannelSchweighoferCah::initialize(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert(V->size()==size);
  assert(dimensions->size()==size);
  assert(Ca_IC->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (m.size()!=size) m.increaseSizeTo(size);
  if (E_Ca.size()!=size) E_Ca.increaseSizeTo(size);
  if (I_Ca.size()!=size) I_Ca.increaseSizeTo(size);

  float gbar_default = gbar[0];
  for (int i=0; i<size; ++i) {
    if (gbar_dists.size() > 0) {
      int j;
      for (j=0; j<gbar_dists.size(); ++j) {
        if ((*dimensions)[0]->dist2soma < gbar_dists[j]) break;
      }
      if (j < gbar_values.size()) 
        gbar[i] = gbar_values[j];
      else
        gbar[i] = gbar_default;
    } else if (gbar_values.size() == 1) {
      gbar[i] = gbar_values[0];
    } else {
      gbar[i] = gbar_default;
    }
  }
  for (unsigned i=0; i<size; ++i) {
    float v=(*V)[i];
    float am = AMC/(1.0 + exp(-(v + AMV)/AMD));
    float bm = BMC*vtrap(v + BMV, BMD);
    m[i] = am/(am + bm);
    g[i] = gbar[i]*m[i]*m[i];
  }
  assert (getSharedMembers().T!=0 && getSharedMembers().Ca_EC!=0);
}

ChannelSchweighoferCah::~ChannelSchweighoferCah()
{
}

