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
#include "NaChannel.h"
#include "CG_NaChannel.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "NumberUtils.h"

#define SMALL 1.0E-6

// There are two models are defined
// Na channel from Hodgkin-Huxley (1952)
//                 Schweighofer-Doya-Kawato (1994) based on Rush-Rinzel (1994)
#define HH_1952
//#define SCHWEIGHOFER_1999

#ifdef HH_1952
#define AMC 0.1
#define AMV 40.0
#define AMD 10.0
#define BMC 4.0
#define BMV 65.0
#define BMD 18.0
#define AHC 0.07
#define AHV 65.0
#define AHD 20.0
#define BHC 1.0
#define BHV 35.0
#define BHD 10.0
#endif
#ifdef SCHWEIGHOFER_1999
#define AMC 0.1
#define AMV 41.0
#define AMD 10.0
#define BMC 9.0
#define BMV 66.0
#define BMD 20.0
// This is 5/170 to account for the 170 in \tau_h
#define AHC 0.029411764705882
#define AHV 60.0
#define AHD 15.0
// This is 1/170 to account for the 170 in \tau_h
#define BHC 0.005882352941176
#define BHV 50.0
#define BHD 10.0
#endif 


void NaChannel::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
    dyn_var_t v=(*V)[i];
    dyn_var_t am = AMC*vtrap(-(v + AMV), AMD);
    dyn_var_t bm = BMC*exp(-(v + BMV)/BMD);
    dyn_var_t ah = AHC*exp(-(v + AHV)/AHD);
#ifdef HH_1952
    dyn_var_t bh = BHC/(1.0 + exp(-(v + BHV)/BHD));
#endif
#ifdef SCHWEIGHOFER_1999
    dyn_var_t bh = BHC*vtrap(-(v + BHV), BHD);
#endif

#ifdef HH_1952
    dyn_var_t pm = 0.5*dt*(am + bm);
    m[i] = (dt*am + m[i]*(1.0 - pm))/(1.0 + pm);
#endif
#ifdef SCHWEIGHOFER_1999
		//TUAN: NOTE Bug in the implementation
    m[i] = am/(am + bm);
#endif
    dyn_var_t ph = 0.5*dt*(ah + bh);
    h[i] = (dt*ah + h[i]*(1.0 - ph))/(1.0 + ph);
    g[i] = gbar[i]*m[i]*m[i]*m[i]*h[i];
  }
}

void NaChannel::initializeNaChannels(RNG& rng)
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
  // allocate 
  if (g.size()!=size) g.increaseSizeTo(size);
  if (m.size()!=size) m.increaseSizeTo(size);
  if (h.size()!=size) h.increaseSizeTo(size);
  // initialize
  for (unsigned i=0; i<size; ++i) {
    gbar[i]=gbar[0];
    dyn_var_t v=(*V)[i];
    dyn_var_t am = AMC*vtrap(-(v + AMV), AMD);
    dyn_var_t bm = BMC*exp(-(v + BMV)/BMD);
    dyn_var_t ah = AHC*exp(-(v + AHV)/AHD);
#ifdef HH_1952
    dyn_var_t bh = BHC/(1.0 + exp(-(v + BHV)/BHD));
#endif
#ifdef SCHWEIGHOFER_1999
    dyn_var_t bh = BHC*vtrap(-(v + BHV), BHD);
#endif
    m[i] = am/(am + bm);
    h[i] = ah/(ah + bh);
    g[i]=gbar[i]*m[i]*m[i]*m[i]*h[i];
  }
}

NaChannel::~NaChannel()
{
}

