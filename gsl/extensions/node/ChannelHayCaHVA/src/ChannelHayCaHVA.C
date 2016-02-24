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
#include "ChannelHayCaHVA.h"
#include "CG_ChannelHayCaHVA.h"
#include "rndm.h"

#define SMALL 1.0E-6

#define AMC -0.055
#define AMV 27
#define AMD -3.8
#define BMC 0.94
#define BMV 75
#define BMD -17
#define AHC 0.000457
#define AHV 13
#define AHD -50
#define BHC 0.0065
#define BHV 15
#define BHD -28

float ChannelHayCaHVA::vtrap(float x, float y) {
  return(fabs(x/y) < SMALL ? y*(1 - x/y/2) : x/(exp(x/y) - 1));
}

void ChannelHayCaHVA::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    E_Ca[i]=(0.04343 * *(getSharedMembers().T) * log(*(getSharedMembers().Ca_EC) / (*Ca_IC)[i]));
    float v=(*V)[i];
    float am = AMC*vtrap(v + AMV, AMD);
    float bm = BMC*exp((v + BMV)/BMD);
    float pm = 0.5*dt*(am + bm);
    m[i] = (dt*am + m[i]*(1.0 - pm))/(1.0 + pm);
    float ah = AHC*exp((v + AHV)/AHD);
    float bh = BHC/(1.0 + exp((v + BHV)/BHD));
    float ph = 0.5*dt*(ah + bh);
    h[i] = (dt*ah + h[i]*(1.0 - ph))/(1.0 + ph);    
    g[i]=gbar[i]*m[i]*m[i]*h[i];
    I_Ca[i] = g[i] * (v-E_Ca[i]);
  }
}

void ChannelHayCaHVA::initialize(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert(V->size()==size);
  assert(dimensions->size()==size);
  assert(Ca_IC->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (m.size()!=size) m.increaseSizeTo(size);
  if (h.size()!=size) h.increaseSizeTo(size);
  if (E_Ca.size()!=size) E_Ca.increaseSizeTo(size);
  if (I_Ca.size()!=size) I_Ca.increaseSizeTo(size);

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
    float bm = BMC*exp((v + BMV)/BMD);
    m[i] = am/(am + bm);
    float ah = AHC*exp(-(v + AHV)/AHD);
    float bh = BHC/(1.0 + exp(-(v + BHV)/BHD));
    h[i] = ah/(ah + bh);
    g[i] = gbar[i]*m[i]*m[i]*h[i];
  }
  assert (getSharedMembers().T!=0 && getSharedMembers().Ca_EC!=0);
}

ChannelHayCaHVA::~ChannelHayCaHVA()
{
}

