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
#include "ChannelSchweighoferNat.h"
#include "CG_ChannelSchweighoferNat.h"
#include "rndm.h"

#define SMALL 1.0E-6

#define AMC 0.1
#define AMV 41.0
#define AMD 10.0
#define BMC 9.0
#define BMV 66.0
#define BMD 20.0
#define AHC 0.029411764705882 // This is 5/170 to account for the 170 in \tau_h
#define AHV 60.0
#define AHD 15.0
#define BHC 0.005882352941176 // This is 1/170 to account for the 170 in \tau_h
#define BHV 50.0
#define BHD 10.0

float ChannelSchweighoferNat::vtrap(float x, float y) {
  return(fabs(x/y) < SMALL ? y*(1 - x/y/2) : x/(exp(x/y) - 1));
}

void ChannelSchweighoferNat::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    float v=(*V)[i];
    float am = AMC*vtrap(-(v + AMV), AMD);
    float bm = BMC*exp(-(v + BMV)/BMD);
    m[i] = am/(am + bm);
    float ah = AHC*exp(-(v + AHV)/AHD);
    float bh = BHC*vtrap(-(v + BHV), BHD);
    float ph = 0.5*dt*(ah + bh);
    h[i] = (dt*ah + h[i]*(1.0 - ph))/(1.0 + ph);
    g[i] = gbar[i]*m[i]*m[i]*m[i]*h[i];
  }
}

void ChannelSchweighoferNat::initialize(RNG& rng)
{
  assert(branchData);
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert(V->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (m.size()!=size) m.increaseSizeTo(size);
  if (h.size()!=size) h.increaseSizeTo(size);
  for (int i=0; i<size; ++i) {
    gbar[i]=gbar[0];
  }
  for (unsigned i=0; i<size; ++i) {
    float v=(*V)[i];
    float am = AMC*vtrap(-(v + AMV), AMD);
    float bm = BMC*exp(-(v + BMV)/BMD);
    m[i] = am/(am + bm);
    float ah = AHC*exp(-(v + AHV)/AHD);
    float bh = BHC*vtrap(-(v + BHV), BHD);
    h[i] = ah/(ah + bh);
    g[i]=gbar[i]*m[i]*m[i]*m[i]*h[i];
  }
}

ChannelSchweighoferNat::~ChannelSchweighoferNat()
{
}
