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
#include "ChannelHayNat.h"
#include "CG_ChannelHayNat.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "MaxComputeOrder.h"
#include "NumberUtils.h"

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

#define AMC 0.182
#define AMV -38.0
#define AMD 6.0
#define BMC -0.124
#define BMV -38.0
#define BMD -6.0
#define AHC -0.015
#define AHV -66.0
#define AHD -6.0
#define BHC 0.015
#define BHV -66.0
#define BHD 6.0
#define T_ADJ 2.9529 // 2.3^((34-21)/10)


void ChannelHayNat::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    dyn_var_t v=(*V)[i];
    //dyn_var_t am = AMC*vtrap(v - AMV, AMD);
    dyn_var_t am = AMC*vtrap(-(v - Vhalf_m[i]), AMD);
    dyn_var_t bm = BMC*vtrap(-(v - BMV), BMD);
    dyn_var_t pm = 0.5*dt*(am + bm) * T_ADJ;
    m[i] = (dt*am*T_ADJ + m[i]*(1.0 - pm))/(1.0 + pm);
    if (m[i] < 0.0) {
      m[i] = 0.0;
    } else if (m[i] > 1.0) {
      m[i] = 1.0;
    }
    dyn_var_t ah = AHC*vtrap(-(v - AHV), AHD);
    dyn_var_t bh = BHC*vtrap(-(v - BHV), BHD);
    dyn_var_t ph = 0.5*dt*(ah + bh)*T_ADJ; 
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
#ifdef DEBUG_ASSERT
  assert(branchData);
#endif
  unsigned size=branchData->size;
#ifdef DEBUG_ASSERT
  assert(V);
  assert(gbar.size()==size);
  assert(V->size()==size);
#endif
  if (g.size()!=size) g.increaseSizeTo(size);
  if (m.size()!=size) m.increaseSizeTo(size);
  if (h.size()!=size) h.increaseSizeTo(size);
	if (Vhalf_m.size() !=size) Vhalf_m.increaseSizeTo(size);
	SegmentDescriptor segmentDescriptor;
  for (unsigned int i=0; i<size; ++i) {
    gbar[i]=gbar[0];
		//NOTE: Shift to the left V1/2 for Nat in AIS region
		Vhalf_m[i] = -31.0; //[mV]
		//Vhalf_m[i] = -38.0; //[mV]
#define DIST_START_AIS   30.0 //[um]
		if ((segmentDescriptor.getBranchType(branchData->key) == Branch::_AIS) 
		//		or 
		//	(		(segmentDescriptor.getBranchType(branchData->key) == Branch::_AXON)  and
	  //		 	(*dimensions)[i]->dist2soma >= DIST_START_AIS)
			)
		{
			//gbar[i] = gbar[i] * 1.50; // increase 3x
			Vhalf_m[i] -= 8.0 ; //[mV]
		}
  }
  for (unsigned i=0; i<size; ++i) {
    dyn_var_t v=(*V)[i];
    //dyn_var_t am = AMC*vtrap(v - AMV, AMD);
    dyn_var_t am = AMC*vtrap(-(v - Vhalf_m[i]), AMD);
    dyn_var_t bm = BMC*vtrap(-(v - BMV), BMD);
    m[i] = am/(am + bm);
    dyn_var_t ah = AHC*vtrap(-(v - AHV), AHD);
    dyn_var_t bh = BHC*vtrap(-(v - BHV), BHD);
    h[i] = ah/(ah + bh);
    g[i]=gbar[i]*m[i]*m[i]*m[i]*h[i];
  }
}

ChannelHayNat::~ChannelHayNat()
{
}

