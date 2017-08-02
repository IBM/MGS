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
#include "ChannelHayNap.h"
#include "CG_ChannelHayNap.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#define SMALL 1.0E-6

// 
// Channel model details for "ChannelHayNap"
// 
// This is an implementation of the "Persistent Na^+ current, I_Nap".
//
// Taken from Hay et al. (2011) "Models of Neocortical Layer 5b Pyramidal Cells..."
// which in turn references the work of Magistretti et al. (1999).
// 
// This has been implemented without modification. 
//

#define IMV 52.6
#define IMD -4.6
#define IHV 48.8
#define IHD 10.0
#define AMC -0.182
#define AMV 38.0
#define AMD -6.0
#define BMC 0.124
#define BMV 38.0
#define BMD 6.0
#define AHC 2.88E-6
#define AHV 17.0
#define AHD 4.63
#define BHC -6.94E-6
#define BHV 64.4
#define BHD -2.63
#define T_ADJ 2.9529 // 2.3^((34-21)/10)

dyn_var_t ChannelHayNap::vtrap(dyn_var_t x, dyn_var_t y) {
  return(fabs(x/y) < SMALL ? y*(1 - x/y/2) : x/(exp(x/y) - 1));
}

void ChannelHayNap::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    dyn_var_t v=(*V)[i];
    dyn_var_t minf = 1/(1+exp((v + IMV)/IMD));
    dyn_var_t am = AMC*vtrap(v + AMV, AMD);
    dyn_var_t bm = BMC*vtrap(v + BMV, BMD);
    dyn_var_t pm = 0.5*dt*(am + bm)*T_ADJ/6.0;
    m[i] = (2*pm*minf + m[i]*(1.0 - pm))/(1.0 + pm);
    dyn_var_t hinf = 1/(1+exp((v + IHV)/IHD));
    dyn_var_t ah = AHC*vtrap(v + AHV, AHD);
    dyn_var_t bh = BHC*vtrap(v + BHV, BHD);
    dyn_var_t ph = 0.5*dt*(ah + bh)*T_ADJ; 
    h[i] = (2*ph*hinf + h[i]*(1.0 - ph))/(1.0 + ph);
    g[i] = gbar[i]*m[i]*m[i]*m[i]*h[i];
#ifdef WAIT_FOR_REST
		float currentTime = getSimulation().getIteration() * (*getSharedMembers().deltaT);
		if (currentTime < NOGATING_TIME)
			g[i]= 0.0;
#endif
  }
}

void ChannelHayNap::initialize(RNG& rng)
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
    dyn_var_t v=(*V)[i];
    m[i] = 1/(1+exp((v + IMV)/IMD));
    h[i] = 1/(1+exp((v + IHV)/IHD));
    g[i]=gbar[i]*m[i]*m[i]*m[i]*h[i];
  }
}

ChannelHayNap::~ChannelHayNap()
{
}

