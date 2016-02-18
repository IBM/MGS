#include "Lens.h"
#include "ChannelKAs.h"
#include "CG_ChannelKAs.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>
static pthread_once_t once_KAs = PTHREAD_ONCE_INIT;

//
// This is an implementation of the "KAs potassium current
//
#if CHANNEL_KAs == KAs_WOLF_2005
// minf(Vm) = 1/(1+exp((Vm-Vh)/k))
// hinf(Vm) = 1/(1+exp(Vm-Vh)/k)
#define VHALF_M -27.0
#define k_M -16.0
#define VHALF_H -33.5
#define k_H 21.5
#define frac_inact 0.996  // 'a' term
#define AHC 1.0
#define AHV 90.96
#define AHD 29.01
#define BHC 1.0
#define BHV 90.96
#define BHD 100
#else
NOT IMPLEMENTED YET
#endif

dyn_var_t ChannelKAs::vtrap(dyn_var_t x, dyn_var_t y)
{
  return (fabs(x / y) < SMALL ? y * (1 - x / y / 2) : x / (exp(x / y) - 1));
}

void ChannelKAs::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_KAs == KAs_WOLF_2005
    // NOTE: Some models use m_inf and tau_m to estimate m
    dyn_var_t tau_m = 0.378 + 9.91 * exp(-pow((v + 34.3) / 30.1, 2));
    dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m * 2);
    dyn_var_t h_a = AHC * exp(-(v + AHV) / AHD);
    dyn_var_t h_b = BHC * exp(-(v + BHV) / BHD);
    dyn_var_t tau_h = 1097.4 / (h_a + h_b);
    dyn_var_t qh = dt * getSharedMembers().Tadj / (tau_h * 2);

    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
#else
    NOT IMPLEMENTED YET
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
    g[i] = gbar[i] * m[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
  }
}

void ChannelKAs::initialize(RNG& rng)
{
  pthread_once(&once_KAs, ChannelKAs::initialize_others);
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
  if (m.size() != size) m.increaseSizeTo(size);
  if (h.size() != size) h.increaseSizeTo(size);
  // initialize
  float gbar_default = gbar[0];
	if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
	{
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels Param"
			<< std::endl;
		assert(0);
	}
  for (unsigned i = 0; i < size; ++i)
  {
    if (gbar_dists.size() > 0) {
      int j;
      assert(gbar_values.size() == gbar_dists.size());
      for (j=0; j<gbar_dists.size(); ++j) {
        if ((*dimensions)[_cptIndex]->dist2soma < gbar_dists[j]) break;
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
      int j;
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
#if CHANNEL_KAs == KAs_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
#else
    NOT IMPLEMENTED YET
// m[i] = am / (am + bm); //steady-state value
// h[i] = ah / (ah + bh);
#endif
    g[i] = gbar[i] * m[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
  }
}

void ChannelKAs::initialize_others() {}

void ChannelKAs::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ChannelKAsInAttrPSet* CG_inAttrPset, CG_ChannelKAsOutAttrPSet* CG_outAttrPset)
{
  _cptIndex=CG_inAttrPset->idx;
}
ChannelKAs::~ChannelKAs() {}
