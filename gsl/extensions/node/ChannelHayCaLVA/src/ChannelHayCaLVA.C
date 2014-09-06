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
#include "ChannelHayCaLVA.h"
#include "CG_ChannelHayCaLVA.h"
#include "rndm.h"

#define SMALL 1.0E-6

#define IMV 40.0
#define IMD -6.0
#define IHV 90.0
#define IHD 6.4
#define TMC 5.0
#define TMF 20.0
#define TMV 35.0
#define TMD 5.0
#define THC 20.0
#define THF 50.0
#define THV 50.0
#define THD 7.0
#define T_ADJ 2.9529 // 2.3^((34-21)/10)

float ChannelHayCaLVA::vtrap(float x, float y) {
  return(fabs(x/y) < SMALL ? y*(x/y/2 - 1) : x/(1 - exp(x/y)));
}

void ChannelHayCaLVA::update(RNG& rng)
{
  float dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    E_Ca[i]=(0.04343 * *(getSharedMembers().T) * log(*(getSharedMembers().Ca_EC) / (*Ca_IC)[i]));
    float v=(*V)[i];
    float minf = 1.0/(1.0 + exp((v + IMV)/IMD));
    float taum = (TMC + TMF/(1+exp((v + TMV)/TMD)))/T_ADJ;
    float hinf = 1.0/(1.0 + exp((v + IHV)/IHD));
    float tauh = (THC + THF/(1+exp((v + THV)/THD)))/T_ADJ;
    float pm = 0.5*dt/taum;
    float ph = 0.5*dt/tauh;
    m[i] = (2.0*pm*minf + m[i]*(1.0 - pm))/(1.0 + pm);
    h[i] = (2.0*ph*hinf + h[i]*(1.0 - ph))/(1.0 + ph);
    g[i] = gbar[i]*m[i]*m[i]*h[i];
    I_Ca[i] = g[i] * (v-E_Ca[i]);
  }
}

void ChannelHayCaLVA::initialize(RNG& rng)
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
    m[i] = 1.0/(1.0 + exp(-(v + IMV)/IMD));
    h[i] = 1.0/(1.0 + exp( (v + IHV)/IHD));
    g[i] = gbar[i]*m[i]*m[i]*h[i];
  }
  assert (getSharedMembers().T!=0 && getSharedMembers().Ca_EC!=0);
}

ChannelHayCaLVA::~ChannelHayCaLVA()
{
}

