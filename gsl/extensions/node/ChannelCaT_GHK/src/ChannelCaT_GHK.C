#include "Lens.h"
#include "ChannelCaT_GHK.h"
#include "CG_ChannelCaT_GHK.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

static pthread_once_t once_CaT_GHK = PTHREAD_ONCE_INIT;

// This is an implementation of T-type Ca2+ channel
//              CaT_GHK current
//
#if CHANNEL_CaT == CaT_GHK_WOLF_2005
// data from  ???
//
// minf(Vm) = 1/(1+exp((Vm-Vh)/k))
// hinf(Vm) = 1/(1+exp(Vm-Vh)/k)
#define VHALF_M -51.73
#define k_M -6.53
#define VHALF_H -80
#define k_H 6.7
#define LOOKUP_TAUM_LENGTH 16  // size of the below array
const dyn_var_t ChannelCaT_GHK::_Vmrange_taum[] = {
    -65, -60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10};
// NOTE:
// if (-100+(-90))/2 >= Vm               : tau_m = taumCaT[1st-element]
// if (-100+(-90))/2 < Vm < (-90+(-80))/2: tau_m = taumCaT[2nd-element]
//...
dyn_var_t ChannelCaT_GHK::taumCaT[] = {20.2, 20.2, 13.1, 8.7, 6.8, 5.6, 4.4, 3.8,
                                   3.6,  3.3,  3.6,  3.6, 3.3, 3.3, 3.3, 3.3};
#define LOOKUP_TAUH_LENGTH 16  // size of the below array
// dyn_var_t _Vmrange_tauh[] = _Vmrange_taum;
const dyn_var_t ChannelCaT_GHK::_Vmrange_tauh[] = {
    -65, -60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10};
dyn_var_t ChannelCaT_GHK::tauhCaT[] = {
    382, 208, 162, 129, 119, 107, 107, 107,
    108, 109, 109, 110, 110, 110, 110, 110,
};
std::vector<dyn_var_t> ChannelCaT_GHK::Vmrange_taum;
std::vector<dyn_var_t> ChannelCaT_GHK::Vmrange_tauh;
#else
NOT IMPLEMENTED YET
#endif

dyn_var_t ChannelCaT_GHK::vtrap(dyn_var_t x, dyn_var_t y)
{
  return (fabs(x / y) < SMALL ? y * (1 - x / y / 2) : x / (exp(x / y) - 1));
}

void ChannelCaT_GHK::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_CaT == CaT_WOLF_2005
    // NOTE: Some models use m_inf and tau_m to estimate m
    // tau_m in the lookup table
    std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
    int index = low - Vmrange_taum.begin();
    //-->tau_m[i] = taumCaT[index];
    // NOTE: dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m[i] * 2);
    dyn_var_t qm = dt * getSharedMembers().Tadj / (taumCaT[index] * 2);
    /* no need to search as they both use the same Vmrange
     * IF NOT< make sure you add this code
    std::vector<dyn_var_t>::iterator low= std::lower_bound(Vmrange_tauh.begin(),
    Vmrange_tauh.end(), v);
    int index = low-Vmrange_tauh.begin();
    */
    dyn_var_t qh = dt * getSharedMembers().Tadj / (tauhCaT[index] * 2);

    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
    //E_Ca[i] = (0.04343 * *(getSharedMembers().T) *
    //           log(*(getSharedMembers().Ca_EC) / (*Ca_IC)[i]));
    dyn_var_t v = (*V)[i];
    PCa[i] = PCabar[i] * m[i] * m[i] * m[i] * h[i];
#endif
		dyn_var_t tmp = exp(-v * zCaF_R / (*getSharedMembers().T));
		//NOTE: PCa [um/ms], Vm [mV], Cai/o [mM], F [C/mol] or [mJ/(mV.mol)]
		//     R [mJ/(mol.K)]
		const dyn_var_t unit_scale = 1e+3; // to convert from nA/um^2 to pA/um^2
    I_Ca[i] = unit_scale * PCa[i] * zCa2F2_R / (*(getSharedMembers().T)) * 
			v * ((*Ca_IC)[i] - *(getSharedMembers().Ca_EC) * tmp)/
			(1- tmp); // [pA/um^2]
  }
}

void ChannelCaT_GHK::initialize(RNG& rng)
{
  pthread_once(&once_CaT_GHK, initialize_others);
#ifdef DEBUG_ASSERT
  assert(branchData);
#endif
  unsigned size = branchData->size;
#ifdef DEBUG_ASSERT
  assert(V);
  assert(PCabar.size() == size);
  assert(V->size() == size);
#endif
  // allocate
  if (m.size() != size) m.increaseSizeTo(size);
  if (h.size() != size) h.increaseSizeTo(size);
  if (PCa.size() != size) PCa.increaseSizeTo(size);
  // initialize
  dyn_var_t PCabar_default = PCabar[0];
  if (Pbar_dists.size() > 0 and Pbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either Pbar_dists or Pbar_branchorders on GHK-formula Ca2+ channel "
                 "Channels Param" << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
    if (Pbar_dists.size() > 0)
    {
      int j;
			assert(Pbar_values.size() == Pbar_dists.size());
      for (j = 0; j < Pbar_dists.size(); ++j)
      {
        if ((*dimensions)[_cptindex]->dist2soma < Pbar_dists[j]) break;
      }
      if (j < Pbar_values.size())
        PCabar[i] = Pbar_values[j];
      else
        PCabar[i] = PCabar_default;
    }
    /*else if (Pbar_values.size() == 1) {
PCabar[i] = Pbar_values[0];
} */
    else if (Pbar_branchorders.size() > 0)
    {
      int j;
			assert(Pbar_values.size() == Pbar_branchorders.size());
      SegmentDescriptor segmentDescriptor;
		  //const special_value = -1; 
      for (j = 0; j < Pbar_branchorders.size(); ++j)
      {
        if (segmentDescriptor.getBranchOrder(branchData->key) ==
            Pbar_branchorders[j])
          break;
			}
			//if (j == Pbar_branchorders.size() and Pbar_branchorders[j-1] == special_value)
			if (j == Pbar_branchorders.size() and Pbar_branchorders[j-1] == GlobalNTS::anybranch_at_end)
			{
				PCabar[i] = Pbar_values[j-1];
			}
			else if (j < Pbar_values.size())
        PCabar[i] = Pbar_values[j];
      else
        PCabar[i] = PCabar_default;
    }
    else
    {
      PCabar[i] = PCabar_default;
    }
  }
  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_CaT == CaT_GHK_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));  // steady-state values
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
#else
    NOT IMPLEMENTED YET
#endif
    PCa[i] = PCabar[i] * m[i] * m[i] * m[i] * h[i];
  }
}

void ChannelCaT_GHK::initialize_others()
{
#if CHANNEL_CaT == CaT_GHK_WOLF_2005
  {
    std::vector<dyn_var_t> tmp(_Vmrange_taum,
                               _Vmrange_taum + LOOKUP_TAUM_LENGTH);
    assert((sizeof(taumCaT) / sizeof(taumCaT[0])) == tmp.size());
    for (int i = 1; i < tmp.size() - 1; i++)
      Vmrange_taum[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
  }
  {
    std::vector<dyn_var_t> tmp(_Vmrange_tauh,
                               _Vmrange_tauh + LOOKUP_TAUH_LENGTH);
    assert(sizeof(tauhCaT) / sizeof(tauhCaT[0]) == tmp.size());
    for (int i = 1; i < tmp.size() - 1; i++)
      Vmrange_tauh[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
  }
#endif
}

void ChannelCaT_GHK::setPointers(const String& CG_direction,
                             const String& CG_component,
                             NodeDescriptor* CG_node, Edge* CG_edge,
                             VariableDescriptor* CG_variable,
                             Constant* CG_constant,
                             CG_ChannelCaT_GHKInAttrPSet* CG_inAttrPset,
                             CG_ChannelCaT_GHKOutAttrPSet* CG_outAttrPset)
{
  _cptindex = CG_inAttrPset->idx;
}

ChannelCaT_GHK::~ChannelCaT_GHK() {}
