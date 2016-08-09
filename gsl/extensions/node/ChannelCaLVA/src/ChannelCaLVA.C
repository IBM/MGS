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
#include "ChannelCaLVA.h"
#include "CG_ChannelCaLVA.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"
#include "MaxComputeOrder.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

#if CHANNEL_CaLVA == CaLVA_HAY_2011
#define IMV -40.0
#define IMD -6.0
#define IHV -90.0
#define IHD 6.4
#define TMC 5.0
#define TMF 20.0
#define TMV 35.0
#define TMD 5.0
#define THC 20.0
#define THF 50.0
#define THV 50.0
#define THD 7.0
//#define T_ADJ 2.9529 // 2.3^((34-21)/10)
//#define T_ADJ 4.17 // 3.0^((34-21)/10)
#else
  NOT IMPLEMENTED YET
#endif

//#define Vhalf_shift 7.0  // [mV]

dyn_var_t ChannelCaLVA::vtrap(dyn_var_t x, dyn_var_t y) {
  return(fabs(x/y) < SMALL ? y*(x/y/2 - 1) : x/(1 - exp(x/y)));
}

void ChannelCaLVA::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    dyn_var_t v = (*V)[i];
    dyn_var_t cai = (*Ca_IC)[i];
#if CHANNEL_CaLVA == CaLVA_HAY_2011
    //E_Ca[i]=(0.04343 * *(getSharedMembers().T) * log(*(getSharedMembers().Ca_EC) / (*Ca_IC)[i]));
    E_Ca[i]=(R_zCaF * *(getSharedMembers().T) * log(*(getSharedMembers().Ca_EC) / cai));
    dyn_var_t minf = 1.0/(1.0 + exp((v - IMV )/IMD));
    //dyn_var_t taum = (TMC + TMF/(1+exp((v + TMV)/TMD)))/T_ADJ;
    dyn_var_t taum = (TMC + TMF/(1+exp((v + TMV)/TMD)))/getSharedMembers().Tadj;
    dyn_var_t hinf = 1.0/(1.0 + exp((v - IHV )/IHD));
    //dyn_var_t tauh = (THC + THF/(1+exp((v + THV)/THD)))/T_ADJ;
    dyn_var_t tauh = (THC + THF/(1+exp((v + THV)/THD)))/getSharedMembers().Tadj;
    dyn_var_t pm = 0.5*dt/taum;
    dyn_var_t ph = 0.5*dt/tauh;
    m[i] = (2.0*pm*minf + m[i]*(1.0 - pm))/(1.0 + pm);
    h[i] = (2.0*ph*hinf + h[i]*(1.0 - ph))/(1.0 + ph);
    g[i] = gbar[i]*m[i]*m[i]*h[i];
#endif

    I_Ca[i] = g[i] * (v-E_Ca[i]);
#ifdef WAIT_FOR_REST
		float currentTime = getSimulation().getIteration() * (*getSharedMembers().deltaT);
		if (currentTime < NOGATING_TIME)
			I_Ca[i] = 0.0;
#endif
    Iion[i] = I_Ca[i]; //g[i] * (v-E_Ca[i]);
  }
}

void ChannelCaLVA::initialize(RNG& rng)
{
  assert (getSharedMembers().T!=0 && getSharedMembers().Ca_EC!=0);
  unsigned size=branchData->size;
  assert(V);
  assert(Ca_IC);
  assert(gbar.size()==size);
  assert(V->size()==size);
  assert(Ca_IC->size()==size);
  // allocate
  if (g.size()!=size) g.increaseSizeTo(size);
  if (m.size()!=size) m.increaseSizeTo(size);
  if (h.size()!=size) h.increaseSizeTo(size);
  if (E_Ca.size()!=size) E_Ca.increaseSizeTo(size);
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
  if (I_Ca.size()!=size) I_Ca.increaseSizeTo(size);
  
  // initialize
  dyn_var_t gbar_default = gbar[0];
  for (unsigned int i=0; i<size; ++i) {
    if (gbar_dists.size() > 0) {
      unsigned int j;
      //NOTE: 'n' bins are splitted by (n-1) points
      if (gbar_values.size() - 1 != gbar_dists.size())
      {
        std::cerr << "gbar_values.size = " << gbar_values.size() 
          << "; gbar_dists.size = " << gbar_dists.size() << std::endl; 
      }
      assert(gbar_values.size() -1 == gbar_dists.size());
      for (j=0; j<gbar_dists.size(); ++j) {
        if ((*dimensions)[i]->dist2soma < gbar_dists[j]) break;
      }
      gbar[i] = gbar_values[j];
    }
		else if (gbar_branchorders.size() > 0)
		{
      unsigned int j;
      assert(gbar_values.size() == gbar_branchorders.size());
      SegmentDescriptor segmentDescriptor;
      for (j=0; j<gbar_branchorders.size(); ++j) {
        if (segmentDescriptor.getBranchOrder(branchData->key) == gbar_branchorders[j]) break;
      }
			if (j == gbar_branchorders.size() and gbar_branchorders[j-1] == GlobalNTS::anybranch_at_end)
			{
				gbar[i] = gbar_values[j-1];
			}
			else if (j < gbar_values.size()) 
        gbar[i] = gbar_values[j];
      else
        gbar[i] = gbar_default;
		}
		else {
      gbar[i] = gbar_default;
    }
  }
    
  for (unsigned i=0; i<size; ++i) {
    dyn_var_t v=(*V)[i];
    dyn_var_t cai = (*Ca_IC)[i];
#if CHANNEL_CaLVA == CaLVA_HAY_2011
    E_Ca[i]=(R_zCaF * *(getSharedMembers().T) * log(*(getSharedMembers().Ca_EC) / cai));
    m[i] = 1.0/(1.0 + exp(-(v - IMV)/IMD));
    h[i] = 1.0/(1.0 + exp( (v - IHV)/IHD));
    g[i] = gbar[i]*m[i]*m[i]*h[i];
#endif
    I_Ca[i] = g[i] * (v-E_Ca[i]);
    Iion[i] = I_Ca[i]; //g[i] * (v-E_Ca[i]);
  }
}

ChannelCaLVA::~ChannelCaLVA()
{
}

