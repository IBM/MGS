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
#include "ChannelHayHCN.h"
#include "CG_ChannelHayHCN.h"
#include "rndm.h"

#define SMALL 1.0E-6

#define AMC 0.00643
#define AMV 154.9
#define AMD 11.9
#define BMC 0.193
#define BMD 33.1

float ChannelHayHCN::vtrap(float x, float y) {
  return(fabs(x/y) < SMALL ? y*(1 - x/y/2) : x/(exp(x/y) - 1));
}

void ChannelHayHCN::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    float v=(*V)[i];
    float am = AMC*vtrap(v + AMV, AMD);
    float bm = BMC*exp(v/BMD);
    float pm = 0.5*dt*(am + bm);
    m[i] = (dt*am + m[i]*(1.0 - pm))/(1.0 + pm);
    g[i] = gbar[i]*m[i];
  }
}

void ChannelHayHCN::initialize(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert(V->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (m.size()!=size) m.increaseSizeTo(size);
  float gbar_default = gbar[0];
  for (int i=0; i<size; ++i) {
    if (gbar_dists.size() > 0) {
      int j;
      assert(gbar_values.size() == gbar_dists.size());
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
    float am = AMC*vtrap(v + AMV, AMD);
    float bm = BMC*exp(v/BMD);
    m[i] = am/(am + bm);
    g[i] = gbar[i]*m[i];
  }
}

ChannelHayHCN::~ChannelHayHCN()
{
}

