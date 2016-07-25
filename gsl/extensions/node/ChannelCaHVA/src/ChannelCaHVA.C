#include "Lens.h"
#include "ChannelCaHVA.h"
#include "CG_ChannelCaHVA.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"
#include "MaxComputeOrder.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

#if CHANNEL_CaHVA == CaHVA_TRAUB_1994 
// NOTE: vtrap(x,y) = x/(exp(x/y)-1)
// a_m  = AMC*(V - AMV)/( exp( (V - AMV)/AMD ) - 1.0 )
// b_m  = BMC * exp( (V - BMV)/BMD )
// a_h  = AHC * exp( (V - AHV)/AHD )
// b_h  = BHC / (exp( (V - BHV)/BHD ) + 1.0)
#define Erev_Ca 75.0 // [mV]
#define am  (1.6 / (1.0 + exp(-0.072 * (v-65))))
#define bm  (0.02 * (vtrap((v-51.1), 5.0)))
#endif

// NOTE: vtrap(x,y) = x/(exp(x/y)-1)
dyn_var_t ChannelCaHVA::vtrap(dyn_var_t x, dyn_var_t y)
{
  return (fabs(x / y) < SMALL ? y * (1 - x / y / 2) : x / (exp(x / y) - 1));
}

void ChannelCaHVA::initialize(RNG& rng) 
{
#ifdef DEBUG_ASSERT
  assert(branchData);
#endif
  unsigned size = branchData->size;
#ifdef DEBUG_ASSERT
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
#endif
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (s.size() != size) s.increaseSizeTo(size);
  if (I_Ca.size() != size) I_Ca.increaseSizeTo(size);
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
  if (E_Ca.size() != size) E_Ca.increaseSizeTo(size);
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
		if (gbar_dists.size() > 0) 
    {
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

  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_CaHVA == CaHVA_TRAUB_1994
    E_Ca[i] = Erev_Ca ; //[mV]
    s[i] = am / (am + bm);  // steady-state value
    g[i] = gbar[i] * s[i] *  s[i];
    I_Ca[i] = g[i] * (v-E_Ca[i]);
    Iion[i] = g[i] * (v-E_Ca[i]);
#else
    // E_rev  = RT/(zF)ln([Ca]o/[Ca]i)   [mV]
    //E_Ca = 0.08617373 * *(getSharedMembers().T) *
    //       log(*(getSharedMembers().Ca_EC) / *(getSharedMembers().Ca_IC));

#endif
  }
}

void ChannelCaHVA::update(RNG& rng) 
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
    // NOTE: Some models use alpha_m and beta_m to estimate m
    // see Rempe-Chopp (2006)
    dyn_var_t pm = 0.5 * dt * (am + bm);
    s[i] = (dt * am + s[i] * (1.0 - pm)) / (1.0 + pm);
    // trick to keep s in [0, 1]
    if (s[i] < 0.0) { s[i] = 0.0; }
    else if (s[i] > 1.0) { s[i] = 1.0; }
#if CHANNEL_CaHVA == CaHVA_TRAUB_1994
    g[i] = gbar[i] * s[i] *  s[i];
    I_Ca[i] = g[i] * (v-E_Ca[i]);
    Iion[i] = g[i] * (v-E_Ca[i]);
#endif
  }
}

ChannelCaHVA::~ChannelCaHVA() 
{
}

