#include "Lens.h"
#include "ChannelKDR_AIS.h"
#include "CG_ChannelKDR_AIS.h"
#include "rndm.h"

#define SMALL 1.0E-6


#if CHANNEL_KDR_AIS == KDR_AIS_TRAUB_1994
#define ANC 0.03
#define ANV 17.2
#define AND 5
#define BNC 0.45
#define BNV 12
#define BND 40
#endif

dyn_var_t ChannelKDR_AIS::vtrap(dyn_var_t x, dyn_var_t y) {
    return(fabs(x/y) < SMALL ? y*(1 - x/y/2) : x/(exp(x/y) - 1));
}

void ChannelKDR_AIS::update(RNG& rng) 
{
dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) {
#if CHANNEL_KDR_AIS == KDR_AIS_TRAUB_1994
    dyn_var_t v=(*V)[i];
    dyn_var_t an = ANC*vtrap((ANV - v), AND);
    dyn_var_t bn = BNC*exp((BNV - v)/BND);
        // see Rempe-Chomp (2006)
    dyn_var_t pn = 0.5*dt*(an + bn);
    n[i] = (dt*an + n[i]*(1.0 - pn))/(1.0 + pn);
    g[i] = gbar[i]*n[i]*n[i]*n[i]*n[i];
#endif
  }
}

void ChannelKDR_AIS::initialize(RNG& rng) 
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
#if CHANNEL_KDR_AIS == KDR_AIS_TRAUB_1994
    dyn_var_t an = ANC*vtrap((ANV-v), AND);
    dyn_var_t bn = BNC*exp((BNV - v)/BND);
    n[i] = an/(an + bn); // steady-state value
    g[i]=gbar[i]*n[i]*n[i]*n[i]*n[i];
#endif
  }
}

ChannelKDR_AIS::~ChannelKDR_AIS() 
{
}

