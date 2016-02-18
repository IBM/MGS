#include "Lens.h"
#include "ChannelKRP.h"
#include "CG_ChannelKRP.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>
static pthread_once_t once_KRP = PTHREAD_ONCE_INIT;

//
// This is an implementation of the "4-AP resistant persistent 
//                  KRP potassium current
//
#if CHANNEL_KRP == KRP_WOLF_2005
// Model: partial inactivation
// minf(Vm) = 1/(1+exp((Vm-Vh)/k))
// hinf(Vm) = 1/(1+exp(Vm-Vh)/k)
#define VHALF_M -13.5
#define k_M -11.8
#define VHALF_H -54.7
#define k_H 18.6
#define frac_inact 0.7  // 'a' term
#define LOOKUP_TAUM_LENGTH 31
const dyn_var_t ChannelKRP::_Vmrange_taum[] = {
    -100, -95, -90, -85, -80, -75, -70, -65, -60, -55, -50,
    -45,  -40, -35, -30, -25, -20, -15, -10, -5,  0,   5,
    10,   15,  20,  25,  30,  35,  40,  45,  50};
dyn_var_t ChannelKRP::taumKRP[] = {40,   45, 48.8, 55, 64.4, 75, 83.9, 90,
                                   93.5, 95, 95.4, 97, 99.2, 95, 79.7, 60,
                                   44.5, 35, 29.3, 25, 20,   15, 11.6, 10,
                                   9.6,  10, 10.5, 10, 8,    5,  5};
#define LOOKUP_TAUH_LENGTH 31
const dyn_var_t ChannelKRP::_Vmrange_tauh[] = {
    -100, -95, -90, -85, -80, -75, -70, -65, -60, -55, -50,
    -45,  -40, -35, -30, -25, -20, -15, -10, -5,  0,   5,
    10,   15,  20,  25,  30,  35,  40,  45,  50};
dyn_var_t ChannelKRP::tauhKRP[] = {
    7000.0, 7000.0, 7000.0, 7000.0, 7000.0, 7000.0, 7000.0, 7000.0,
    7000.0, 7000.0, 7000.0, 7000.0, 7000.0, 7000.0, 7000.0, 7000.0,
    7000.0, 7000.0, 7000.0, 7000.0, 6742.5, 6000.0, 4740.2, 3500.0,
    2783.3, 2500.0, 2336.3, 2200.0, 2083.5, 2000.0, 2000.0};
std::vector<dyn_var_t> ChannelKRP::Vmrange_taum;
std::vector<dyn_var_t> ChannelKRP::Vmrange_tauh;
#else
NOT IMPLEMENTED YET
// const dyn_var_t _Vmrange_taum[] = {};
// const dyn_var_t _Vmrange_tauh[] = {};
// const dyn_var_t taumKRP[] = {};
// const dyn_var_t tauhKRP[] = {};
#endif

void ChannelKRP::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_KRP == KRP_WOLF_2005
    // NOTE: Some models use m_inf and tau_m to estimate m
    // tau_m in the lookup table
    std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
    int index = low - Vmrange_taum.begin();
    //-->tau_m[i] = taumKRP[index];
    // NOTE: dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m[i] * 2);
    dyn_var_t qm = dt * getSharedMembers().Tadj / (taumKRP[index] * 2);
    /* no need to search as they both use the same Vmrange
     * IF NOT< make sure you add this code
    std::vector<dyn_var_t>::iterator low= std::lower_bound(Vmrange_tauh.begin(),
    Vmrange_tauh.end(), v);
    int index = low-Vmrange_tauh.begin();
    */
    dyn_var_t qh = dt * getSharedMembers().Tadj / (tauhKRP[index] * 2);

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
    g[i] = gbar[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
  }
}

void ChannelKRP::initialize(RNG& rng)
{

  pthread_once(&once_KRP, ChannelKRP::initialize_others);
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
#if CHANNEL_KRP == KRP_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
#else
    NOT IMPLEMENTED YET
#endif
    g[i] = gbar[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
  }
}

void ChannelKRP::initialize_others()
{
#if CHANNEL_KRP == KRP_WOLF_2005
  {
    std::vector<dyn_var_t> tmp(_Vmrange_taum,
                               _Vmrange_taum + LOOKUP_TAUM_LENGTH);
    assert((sizeof(taumKRP) / sizeof(taumKRP[0])) == tmp.size());
    for (int i = 1; i < tmp.size() - 1; i++)
      Vmrange_taum[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
  }
  {
    std::vector<dyn_var_t> tmp(_Vmrange_tauh,
                               _Vmrange_tauh + LOOKUP_TAUH_LENGTH);
    assert(sizeof(tauhKRP) / sizeof(tauhKRP[0]) == tmp.size());
    for (int i = 1; i < tmp.size() - 1; i++)
      Vmrange_tauh[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
  }
#endif
}

void ChannelKRP::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ChannelKRPInAttrPSet* CG_inAttrPset, CG_ChannelKRPOutAttrPSet* CG_outAttrPset)
{
  _cptIndex=CG_inAttrPset->idx;
}
ChannelKRP::~ChannelKRP() {}
