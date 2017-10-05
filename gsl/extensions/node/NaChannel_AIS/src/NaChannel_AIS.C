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
#include "NaChannel_AIS.h"
#include "CG_NaChannel_AIS.h"
#include "ShallowArray.h"
#include "rndm.h"
#include "NumberUtils.h"

#define SMALL 1.0E-6

// Na channel used for the axial initiation site (AIS), i.e. axon hillock
//
#define DEGRUIJL_2012

/* Degruijl 2012 used the following parameters:
   gbar_0 = 120.0;
   gbar_AIS = 240.0;
   hrv_0 = 70.0;
   hrv_AIS = 60.0;
   trc_0 = 3.0;
   trc_AIS = 1.5; */

#define MRV 30.0
#define MRD 5.5
#define HRD -5.8
#define TRV 40.0
#define TRD 33.0


void NaChannel_AIS::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    dyn_var_t v=(*V)[i];
    dyn_var_t minf = 1.0/(1.0 + exp(-(v + MRV) / MRD) );
    dyn_var_t hinf = 1.0/(1.0 + exp(-(v + hrv[i]) / HRD) );
    dyn_var_t tauh = trc[i]*exp( -(v + TRV) / TRD);
    dyn_var_t ph = 0.5*dt/tauh;
    h[i] = (2.0*ph*hinf + h[i]*(1.0 - ph))/(1.0 + ph);
    g[i] = gbar[i]*minf*minf*minf*h[i];
  }
}
  
void NaChannel_AIS::initializeNaChannels(RNG& rng)
{
#ifdef DEBUG_ASSERT
  assert(branchData);
  assert(dimensions);  
#endif
  unsigned size=branchData->size;
#ifdef DEBUG_ASSERT
  assert(V);
  assert(V->size()==size);
#endif
  if (gbar.size()!=size) gbar.increaseSizeTo(size);
  if (hrv.size()!=size) hrv.increaseSizeTo(size);
  if (trc.size()!=size) trc.increaseSizeTo(size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (h.size()!=size) h.increaseSizeTo(size);
  ShallowArray<DimensionStruct*>& dimensionsArray = *dimensions;
  for (unsigned i=0; i<size; ++i) {
    gbar[i]=getSharedMembers().gbar_0;
    hrv[i]=getSharedMembers().hrv_0;
    trc[i]=getSharedMembers().trc_0;
    if (dimensionsArray[i]->dist2soma<=getSharedMembers().d_AIS) {
      gbar[i]=getSharedMembers().gbar_AIS;
      hrv[i]=getSharedMembers().hrv_AIS;
      trc[i]=getSharedMembers().trc_AIS;
    }
    dyn_var_t v=(*V)[i];
    h[i] = 1.0/(1.0 + exp(-(v + hrv[i]) / HRD) );
    dyn_var_t minf = 1.0/(1.0 + exp(-(v + MRV) / MRD) );
    g[i] = gbar[i]*minf*minf*minf*h[i];
  }
}

NaChannel_AIS::~NaChannel_AIS()
{
}

