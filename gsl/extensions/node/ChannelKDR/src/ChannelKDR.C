// =================================================================
// Licensed Materials - Property of IBM
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
#include "ChannelKDR.h"
#include "CG_ChannelKDR.h"
#include "rndm.h"

#define SMALL 1.0E-6


#if CHANNEL_KDR == KDR_HODGKINHUXLEY_1952
// data measured from squid giant axon
#define ANC 0.01
#define ANV 55.0
#define AND 10.0
#define BNC 0.125
#define BNV 65.0
#define BND 80.0
#elif CHANNEL_KDR == KDR_SCHWEIGHOFER_1999
#define ANC 0.2
#define ANV 41.0
#define AND 10.0
#define BNC 2.5
#define BNV 51.0
#define BND 80.0
#elif CHANNEL_KDR == KDR_TRAUB_1994
#define ANC 0.016
#define ANV 35.1
#define AND 5
#define BNC 0.25
#define BNV 20
#define BND 40
#endif

dyn_var_t ChannelKDR::vtrap(dyn_var_t x, dyn_var_t y) {
	return(fabs(x/y) < SMALL ? y*(1 - x/y/2) : x/(exp(x/y) - 1));
}

void ChannelKDR::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
#if CHANNEL_KDR == KDR_HODGKINHUXLEY_1952 || CHANNEL_KDR == KDR_SCHWEIGHOFER_1999
    dyn_var_t v=(*V)[i];
    dyn_var_t an = ANC*vtrap(-(v + ANV), AND);
    dyn_var_t bn = BNC*exp(-(v + BNV)/BND);
		// see Rempe-Chomp (2006)
    dyn_var_t pn = 0.5*dt*(an + bn) * getSharedMembers().Tadj;
    n[i] = (dt*an*getSharedMembers().Tadj + n[i]*(1.0 - pn))/(1.0 + pn);
    g[i] = gbar[i]*n[i]*n[i]*n[i]*n[i];
#elif CHANNEL_KDR == KDR_TRAUB_1994
    dyn_var_t v=(*V)[i];
    dyn_var_t an = ANC*vtrap((ANV - v), AND);
    dyn_var_t bn = BNC*exp((BNV - v)/BND);
		// see Rempe-Chomp (2006)
    dyn_var_t pn = 0.5*dt*(an + bn);
    n[i] = (dt*an + n[i]*(1.0 - pn))/(1.0 + pn);
    g[i] = gbar[i]*n[i]*n[i];
#else
		NOT IMPLEMENTED YET
#endif
  }
}

void ChannelKDR::initialize(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert (V->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (n.size()!=size) n.increaseSizeTo(size);
  for (unsigned i=0; i<size; ++i) {
    gbar[i]=gbar[0];
    dyn_var_t v=(*V)[i];
#if CHANNEL_KDR == KDR_HODGKINHUXLEY_1952 || CHANNEL_KDR == KDR_SCHWEIGHOFER_1999
    dyn_var_t an = ANC*vtrap(-(v + ANV), AND);
    dyn_var_t bn = BNC*exp(-(v + BNV)/BND);
    n[i] = an/(an + bn); // steady-state value
    g[i]=gbar[i]*n[i]*n[i]*n[i]*n[i];
#elif CHANNEL_KDR == KDR_TRAUB_1994
    dyn_var_t an = ANC*vtrap((ANV-v), AND);
    dyn_var_t bn = BNC*exp((BNV - v)/BND);
    n[i] = an/(an + bn); // steady-state value
    g[i]=gbar[i]*n[i]*n[i];
    g[i] = gbar[i]*n[i]*n[i];
#endif
  }
}

ChannelKDR::~ChannelKDR()
{
}

