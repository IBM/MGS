// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "ChannelMK.h"
#include "CG_ChannelMK.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "MaxComputeOrder.h"
#include "GlobalNTSConfig.h"
#define SMALL 1.0E-6

// Muscarinic-activated K+ outward current
//
#if CHANNEL_MK == MK_ADAMS_BROWN_CONSTANTI_1982
#define AMC 0.0033
#define AMV 35.0
#define AMD 0.1
#define BMC 0.0033
#define BMV 35.0
#define BMD -0.1
//#define T_ADJ 2.9529 // 2.3^((34-21)/10)
#else
   NOT IMPLEMENTED YET
#endif
dyn_var_t ChannelMK::vtrap(dyn_var_t x, dyn_var_t y) {
	return(fabs(x/y) < SMALL ? y*(1 - x/y/2) : x/(exp(x/y) - 1));
}

void ChannelMK::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i<branchData->size; ++i) {
    dyn_var_t v = (*V)[i];
#if CHANNEL_MK == MK_ADAMS_BROWN_CONSTANTI_1982
    dyn_var_t am = AMC*exp(AMD*(v + AMV));
    dyn_var_t bm = BMC*exp(BMD*(v + BMV));
    //dyn_var_t pm = 0.5*dt*(am + bm)*T_ADJ; 
    dyn_var_t pm = 0.5*dt*(am + bm)*getSharedMembers().Tadj; 
    //m[i] = (dt*am*T_ADJ + m[i]*(1.0 - pm))/(1.0 + pm);
    m[i] = (dt*am*getSharedMembers().Tadj + m[i]*(1.0 - pm))/(1.0 + pm);
    g[i] = gbar[i]*m[i];
#endif
  }
}

void ChannelMK::initialize(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert(V->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (m.size()!=size) m.increaseSizeTo(size);
  // initialize
  dyn_var_t gbar_default = gbar[0];
	if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
	{
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels Param"
			<< std::endl;
		assert(0);
	}
  for (unsigned i = 0; i < size; ++i)
  {
    if (gbar_dists.size() > 0) {
      unsigned int j;
      assert(gbar_values.size() == gbar_dists.size());
      for (j=0; j<gbar_dists.size(); ++j) {
        if ((*dimensions)[i]->dist2soma < gbar_dists[j]) break;
      }
      if (j < gbar_values.size()) 
        gbar[i] = gbar_values[j];
      else
        gbar[i] = gbar_default;
    } 
		/*else if (gbar_values.size() == 1) {
      gbar[i] = gbar_values[0];
    } */
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
    dyn_var_t v = (*V)[i];
#if CHANNEL_MK == MK_ADAMS_BROWN_CONSTANTI_1982
    dyn_var_t am = AMC*exp(AMD*(v + AMV));
    dyn_var_t bm = BMC*exp(BMD*(v + BMV));
    m[i] = am/(am + bm);
    g[i] = gbar[i]*m[i];
#endif
  }
}

ChannelMK::~ChannelMK()
{
}

