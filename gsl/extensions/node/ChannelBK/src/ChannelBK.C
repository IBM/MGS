#include "Lens.h"
#include "ChannelBK.h"
#include "CG_ChannelBK.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"
#include "MaxComputeOrder.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

#define Cai_base 0.1e-6 // [uM]
//
// Implementation of the KCa potassium current
//  Voltage and Calcium dependent Potassium
#if CHANNEL_BK == BK_TRAUB_1994
// This implementation is from Traub et al. 1994
//  CA1 pyramidal neuron model
//  Used in Traub's models of several neuron types
//  Non-inactivating, V- and Ca-dep K
//  Alpha-Beta forward-backward rate formulation model
#else
    NOT IMPLEMENTED YET
#endif

void ChannelBK::update(RNG& rng) 
{
    dyn_var_t dt = *(getSharedMembers().deltaT);
    for (unsigned i = 0; i < branchData->size; ++i)
    {
        dyn_var_t v = (*V)[i];
#if SIMULATION_INVOLVE == VMONLY
        dyn_var_t cai = Cai_base;            
#else
        dyn_var_t cai = (*Cai)[i]; // [uM]
#endif

#if CHANNEL_BK == BK_TRAUB_1994
        dyn_var_t alpha, beta;
        if ( v <= 50 ) {
            alpha = exp(((v-10)/11) - ((v-6.5)/27))/18.975;
            beta = 2*exp(-((v-6.5)/27))-alpha;
        } else  {
            alpha = 2*exp(-((v-6.5)/27));
            beta = 0.0;
        }
        // Rempe * Chopp (2006)
        dyn_var_t pc = 0.5*dt*(alpha+beta);
        fO[i] = (dt*alpha + fO[i]*(1.0-pc))/(1.0+pc);
        //dyn_var_t CaGate = (cai/250.0)>1.0?1.0:(cai/250.0);
        dyn_var_t CaGate = (cai/0.250)>1.0?1.0:(cai/0.250);
        g[i] = gbar[i]*fO[i]*CaGate;
    }
#endif
}

void ChannelBK::initialize(RNG& rng) 
{
  assert(branchData);
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert (V->size()==size);
  if (fO.size()!=size) fO.increaseSizeTo(size);
  if (g.size()!=size) g.increaseSizeTo(size);

  // initialize
  SegmentDescriptor segmentDescriptor;
  float gbar_default = gbar[0];
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

  for (unsigned i = 0; i < size; ++i) {
#if SIMULATION_INVOLVE == VMONLY
    dyn_var_t cai = Cai_base;            
#else
    assert(Cai);
    dyn_var_t cai = (*Cai)[i]; // [uM]
#endif

#if CHANNEL_BK == BK_TRAUB_1994
    dyn_var_t alpha, beta ;
    dyn_var_t v=(*V)[i];
    if ( v <= 50 ) {
      alpha = exp(((v-10)/11) - ((v-6.5)/27))/18.975;
      beta = 2*exp(-((v-6.5)/27))-alpha;
    } else {
      alpha = 2*exp(-((v-6.5)/27));
      beta = 0.0;
    }
    fO[i] = alpha/(alpha+beta); // steady-state value
    //dyn_var_t CaGate = (cai/250.0)>1.0?1.0:(cai/250.0);
    dyn_var_t CaGate = (cai/0.250)>1.0?1.0:(cai/0.250);
    g[i] = gbar[i]*fO[i]*CaGate;
#endif
  }
}

ChannelBK::~ChannelBK() 
{
}

