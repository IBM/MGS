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
#include "ChannelHayNat.h"
#include "CG_ChannelHayNat.h"
#include "rndm.h"

#define SMALL 1.0E-6

// 
// Channel model details for "ChannelHayNat"
// 
// This is an implementation of the "Fast, inactivating Na^+ current, I_Nat".
//
// Taken from Hay et al. (2011) "Models of Neocortical Layer 5b Pyramidal Cells..."
// which in turn references the work of Colbert et al. (2002).
// 
// This has been implemented without modification. 
//

#define AMC -0.182
#define AMV 38.0
#define AMD -6.0
#define BMC 0.124
#define BMV 38.0
#define BMD 6.0
#define AHC 0.015
#define AHV 66.0
#define AHD 6.0
#define BHC -0.015
#define BHV 66.0
#define BHD -6.0
#define T_ADJ 2.9529 // 2.3^((34-21)/10)

float ChannelHayNat::vtrap(float x, float y) {
  return(fabs(x/y) < SMALL ? y*(1 - x/y/2) : x/(exp(x/y) - 1));
}

void ChannelHayNat::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    float v=(*V)[i];
    float am = AMC*vtrap(v + AMV, AMD);
    float bm = BMC*vtrap(v + BMV, BMD);
    float pm = 0.5*dt*(am + bm);
    m[i] = (dt*am + m[i]*(1.0 - pm))/(1.0 + pm);
    if (m[i] < 0.0) {
      m[i] = 0.0;
    } else if (m[i] > 1.0) {
      m[i] = 1.0;
    }
    float ah = AHC*vtrap(v + AHV, AHD);
    float bh = BHC*vtrap(v + BHV, BHD);
    float ph = 0.5*dt*(ah + bh)*T_ADJ; 
    h[i] = (dt*ah*T_ADJ + h[i]*(1.0 - ph))/(1.0 + ph);
    if (h[i] < 0.0) {
      h[i] = 0.0;
    } else if (h[i] > 1.0) {
      h[i] = 1.0;
    }
    g[i] = gbar[i]*m[i]*m[i]*m[i]*h[i];
  }
}

void ChannelHayNat::initialize(RNG& rng)
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
    float am = AMC*vtrap(v + AMV, AMD);
    float bm = BMC*vtrap(v + BMV, BMD);
    m[i] = am/(am + bm);
    float ah = AHC*vtrap(v + AHV, AHD);
    float bh = BHC*vtrap(v + BHV, BHD);
    h[i] = ah/(ah + bh);
    g[i]=gbar[i]*m[i]*m[i]*m[i]*h[i];
  }
}

ChannelHayNat::~ChannelHayNat()
{
}

