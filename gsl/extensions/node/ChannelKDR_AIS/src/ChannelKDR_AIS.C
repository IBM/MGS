#include "Lens.h"
#include "ChannelKDR_AIS.h"
#include "CG_ChannelKDR_AIS.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"
#include "MaxComputeOrder.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

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
  for (unsigned i=0; i<branchData->size; ++i) 
  {
    dyn_var_t v=(*V)[i];
#if CHANNEL_KDR_AIS == KDR_AIS_TRAUB_1994
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
  assert(branchData);
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert (V->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (n.size()!=size) n.increaseSizeTo(size);
  float gbar_default = gbar[0];
  //initialize
  SegmentDescriptor segmentDescriptor;
  if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels Param"
      << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
    //gbar init
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

