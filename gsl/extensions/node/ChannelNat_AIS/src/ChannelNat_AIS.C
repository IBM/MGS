#include "Lens.h"
#include "ChannelNat_AIS.h"
#include "CG_ChannelNat_AIS.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"
#include "MaxComputeOrder.h"

#define SMALL 1.0E-6

#if CHANNEL_NAT_AIS == NAT_AIS_TRAUB_1994
//Developed for Gamagenesis in interneurons
//All above conventions for a_m, a_h, b_h remain the same as above except b_m below
//b_m = (BMC * (V - BMV))/(exp((V-BMV)/BMD)-1)
#define Eleak -65.0 //[mV]
#define AMC -0.8
#define AMV (17.2+Eleak)
#define AMD -4.0
#define BMC 0.7
#define BMV (42.2+Eleak)
#define BMD 5.0
#define AHC 0.3
#define AHV (-42.0+Eleak)
#define AHD -18.0
#define BHC 10.0
#define BHV (-42.0+Eleak)
#define BHD -5.0
#endif

// NOTE: vtrap(x,y) = x/(exp(x/y)-1)
dyn_var_t ChannelNat_AIS::vtrap(dyn_var_t x, dyn_var_t y)
{
  return (fabs(x / y) < SMALL ? y * (1 - x / y / 2) : x / (exp(x / y) - 1));
}

void ChannelNat_AIS::update(RNG& rng) 
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_NAT_AIS == NAT_AIS_TRAUB_1994
    dyn_var_t am = AMC * vtrap((v - AMV), AMD);
    dyn_var_t bm = (BMC * (v-BMV)) /  (exp((v - BMV) / BMD) - 1);
    dyn_var_t ah = AHC * exp((v - AHV) / AHD);
    dyn_var_t bh = BHC / (1.0 + exp((v - BHV) / BHD));
    // Traub Models do not have temperature dependence and hence Tadj is not used
    dyn_var_t pm = 0.5 * dt * (am + bm) ;
    m[i] = (dt * am  + m[i] * (1.0 - pm)) / (1.0 + pm);
    dyn_var_t ph = 0.5 * dt * (ah + bh) ;
    h[i] = (dt * ah  + h[i] * (1.0 - ph)) / (1.0 + ph);
#endif
    // trick to keep m in [0, 1]
    if (m[i] < 0.0) { m[i] = 0.0; }
    else if (m[i] > 1.0) { m[i] = 1.0; }
    // trick to keep m in [0, 1]
    if (h[i] < 0.0)
    {
      h[i] = 0.0;
    }
    else if (h[i] > 1.0)
    {
      h[i] = 1.0;
    }
#if CHANNEL_NAT_AIS == NAT_AIS_TRAUB_1994
    g[i] = gbar[i] * m[i] *  m[i] * m[i] * h[i];
		Iion[i] = g[i] * (v - getSharedMembers().E_Na[0]);
#endif
  }
}


void ChannelNat_AIS::initialize(RNG& rng) 
{
  assert(branchData);
  unsigned size = branchData->size;
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (m.size() != size) m.increaseSizeTo(size);
  if (h.size() != size) h.increaseSizeTo(size);
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
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
#if CHANNEL_NAT_AIS == NAT_AIS_TRAUB_1994    
    dyn_var_t am = AMC * vtrap((v - AMV), AMD);
    dyn_var_t bm = (BMC * (v-BMV)) /  (exp((v - BMV) / BMD) - 1);
    dyn_var_t ah = AHC * exp((v - AHV) / AHD);
    dyn_var_t bh = BHC / (1.0 + exp((v - BHV) / BHD));
    m[i] = am / (am + bm);  // steady-state value
    h[i] = ah / (ah + bh);
    g[i] = gbar[i] * m[i] *  m[i] * m[i] * h[i];
		Iion[i] = g[i] * (v - getSharedMembers().E_Na[0]);
#else
    NOT IMPLEMENTED
#endif

  }
}

ChannelNat_AIS::~ChannelNat_AIS() 
{
}

