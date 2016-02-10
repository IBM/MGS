#include "Lens.h"
#include "ChannelNap.h"
#include "CG_ChannelNap.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

static pthread_once_t once_Nap = PTHREAD_ONCE_INIT;

//
// This is an implementation of the "TTX-sensitive slowly-activating, and
// persistent-activating
//         Vm-gated Na^+ current, I_Nap".
//
#if CHANNEL_NAP == NAP_WOLF_2005
// data
//    activation : from entorhinal cortical stellate cell (Magistretti-Alonso,
//    1999)
//    inactivation: from computational model of Layer2/3 pyramidal neuron
//    (Traub, 2003)
//     recorded at 22-24C and then mapped to 35C using Q10 = 3
//
// model is used for simulation NAc nucleus accumbens (medium-sized spiny MSN
// cell)
//    at 35.0 Celcius
// minf(Vm) = 1/(1+exp((Vm-Vh)/k))
// hinf(Vm) = 1/(1+exp(Vm-Vh)/k)
#define VHALF_M -52.6
#define k_M -4.6
#define VHALF_H -48.8
#define k_H 10.0
#define LOOKUP_TAUH_LENGTH 15  // size of the below array
const dyn_var_t ChannelNap::_Vmrange_tauh[] = {-100, -90, -80, -70, -60, -50, -40, -30,
                                    -20,  -10, 0,   10,  20,  30,  40};
dyn_var_t ChannelNap::tauhNap[] = {4500, 4750, 5200, 6100, 6300, 5000, 4250, 3500,
                              3000, 2700, 2500, 2100, 2100, 2100, 2100};
std::vector<dyn_var_t> ChannelNap::Vmrange_tauh;
#else
NOT IMPLEMENTED YET
#endif

dyn_var_t ChannelNap::vtrap(dyn_var_t x, dyn_var_t y)
{
  return (fabs(x / y) < SMALL ? y * (1 - x / y / 2) : x / (exp(x / y) - 1));
}

void ChannelNap::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_NAP == NAP_WOLF_2005
    // NOTE: Some models use m_inf and tau_m to estimate m
    dyn_var_t tau_m;
    if (v < -40.0)
      tau_m = 0.025 + 0.14 * exp((v + 40) / 10);
    else
      tau_m = 0.02 + 0.145 * exp(-(v + 40) / 10);
    dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m * 2);
    std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_tauh.begin(), Vmrange_tauh.end(), v);
    int index = low - Vmrange_tauh.begin();
    dyn_var_t qh = dt * getSharedMembers().Tadj / (tauhNap[index] * 2);

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
    g[i] = gbar[i] * m[i] * h[i];
  }
}

void ChannelNap::initialize(RNG& rng)
{
  pthread_once(&once_Nap, ChannelNap::initialize_others);
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
	if (gbar_dists.size() > 0 and gbar_branchorder.size() > 0)
	{
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorder on Channels Param"
			<< std::endl;
		assert(0);
	}
  for (unsigned i = 0; i < size; ++i)
  {
    if (gbar_dists.size() > 0) {
      int j;
      for (j=0; j<gbar_dists.size(); ++j) {
        if ((*dimensions)[_cptindex]->dist2soma < gbar_dists[j]) break;
      }
      if (j < gbar_values.size()) 
        gbar[i] = gbar_values[j];
      else
        gbar[i] = gbar_default;
    } 
		/*else if (gbar_values.size() == 1) {
      gbar[i] = gbar_values[0];
    } */
		else if (gbar_branchorder.size() > 0)
		{
      int j;
			SegmentDescriptor segmentDescriptor;
      for (j=0; j<gbar_branchorder.size(); ++j) {
        if (segmentDescriptor.getBranchOrder(branchData->key) == gbar_branchorder[j]) break;
      }
      if (j < gbar_values.size()) 
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
#if CHANNEL_NAP == NAP_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
#else
    NOT IMPLEMENTED YET m[i] = am / (am + bm);  // steady-state value
    h[i] = ah / (ah + bh);
#endif
    g[i] = gbar[i] * m[i] * h[i];
  }
}

void ChannelNap::initialize_others()
{
#if CHANNEL_NAP == NAP_WOLF_2005
  std::vector<dyn_var_t> tmp(_Vmrange_tauh, _Vmrange_tauh + LOOKUP_TAUH_LENGTH);
  assert((sizeof(tauhNap) / sizeof(tauhNap[0])) == tmp.size());
  for (int i = 1; i < tmp.size() - 1; i++)
    Vmrange_tauh[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
#endif
}

void ChannelNap::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ChannelNapInAttrPSet* CG_inAttrPset, CG_ChannelNapOutAttrPSet* CG_outAttrPset)
{
  _cptindex=CG_inAttrPset->idx;
}
ChannelNap::~ChannelNap() {}
