#include "Lens.h"
#include "ChannelCaR_GHK.h"
#include "CG_ChannelCaR_GHK.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>
static pthread_once_t once_CaR_GHK = PTHREAD_ONCE_INIT;

// This is an implementation of R-type Ca2+ channel
//              CaR_GHK current
//
#if CHANNEL_CaR == CaR_GHK_WOLF_2005
// same kinetics as that of CaLv12 of Wolf2005, just Vhalf-activated is lower
// minf(Vm) = 1/(1+exp((Vm-Vh)/k))
// hinf(Vm) = 1/(1+exp(Vm-Vh)/k)
#define VHALF_M -10.3
#define k_M -6.6
#define VHALF_H -33.3
#define k_H 17
#define frac_inact  1

#define LOOKUP_TAUH_LENGTH 6  // size of the below array
const dyn_var_t ChannelCaR_GHK::_Vmrange_tauh[] = {-30, -20, -10, 0, 10, 20};
dyn_var_t ChannelCaR_GHK::tauhCaR[] = {100, 65, 35, 30, 20, 20};
std::vector<dyn_var_t> ChannelCaR_GHK::Vmrange_tauh;
#else
#define frac_inact  1.0
NOT IMPLEMENTED YET
#endif

dyn_var_t ChannelCaR_GHK::vtrap(dyn_var_t x, dyn_var_t y)
{
  return (fabs(x / y) < SMALL ? y * (1 - x / y / 2) : x / (exp(x / y) - 1));
}

void ChannelCaR_GHK::initialize(RNG& rng)
{
  pthread_once(&once_CaR_GHK, initialize_others);
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
    std::cerr << "ERROR: Use either Pbar_dists or Pbar_branchorders on "
                 "GHK-formula Ca2+ channel "
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
      // const special_value = -1;
      for (j = 0; j < Pbar_branchorders.size(); ++j)
      {
        if (segmentDescriptor.getBranchOrder(branchData->key) ==
            Pbar_branchorders[j])
          break;
      }
      // if (j == Pbar_branchorders.size() and Pbar_branchorders[j-1] ==
      // special_value)
      if (j == Pbar_branchorders.size() and
          Pbar_branchorders[j - 1] == GlobalNTS::anybranch_at_end)
      {
        PCabar[i] = Pbar_values[j - 1];
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
#if CHANNEL_CaR == CaR_GHK_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));  // steady-state values
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
    PCa[i] =
        PCabar[i] * m[i] * m[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
    dyn_var_t tmp = exp(-v * zCaF_R / (*getSharedMembers().T));
    // NOTE: PCa [um/ms], Vm [mV], Cai/o [uM], F [C/mol] or [mJ/(mV.mol)]
    //     R [mJ/(mol.K)]
    I_Ca[i] = PCa[i] * zCa2F2_R / (*(getSharedMembers().T)) * v *
              ((*Ca_IC)[i] - *(getSharedMembers().Ca_EC) * tmp) /
              (1 - tmp);  // [pA/um^2]
#else
    NOT IMPLEMENTED YET
#endif
  }
}

void ChannelCaR_GHK::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_CaR == CaR_WOLF_2005
    // NOTE: Some models use m_inf and tau_m to estimate m
    dyn_var_t tau_m = 1.6;  // msec
    dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m * 2);

    std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_tauh.begin(), Vmrange_tauh.end(), v);
    int index = low - Vmrange_tauh.begin();
    dyn_var_t qh = dt * getSharedMembers().Tadj / (tauhCaR[index] * 2);

    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
    // E_Ca[i] = (0.04343 * *(getSharedMembers().T) *
    //           log(*(getSharedMembers().Ca_EC) / (*Ca_IC)[i]));
    PCa[i] =
        PCabar[i] * m[i] * m[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
    dyn_var_t tmp = exp(-v * zCaF_R / (*getSharedMembers().T));
    // NOTE: PCa [um/ms], Vm [mV], Cai/o [uM], F [C/mol] or [mJ/(mV.mol)]
    //     R [mJ/(mol.K)]
    I_Ca[i] = PCa[i] * zCa2F2_R / (*(getSharedMembers().T)) * v *
              ((*Ca_IC)[i] - *(getSharedMembers().Ca_EC) * tmp) /
              (1 - tmp);  // [pA/um^2]
#endif
  }
}

void ChannelCaR_GHK::setPointers(const String& CG_direction,
                                 const String& CG_component,
                                 NodeDescriptor* CG_node, Edge* CG_edge,
                                 VariableDescriptor* CG_variable,
                                 Constant* CG_constant,
                                 CG_ChannelCaR_GHKInAttrPSet* CG_inAttrPset,
                                 CG_ChannelCaR_GHKOutAttrPSet* CG_outAttrPset)
{
  _cptindex = CG_inAttrPset->idx;
}

void ChannelCaR_GHK::initialize_others() 
{
#if CHANNEL_CaR == CaR_WOLF_2005
  {
    std::vector<dyn_var_t> tmp(_Vmrange_tauh,
                               _Vmrange_tauh + LOOKUP_TAUH_LENGTH);
    assert(sizeof(tauhCaR) / sizeof(tauhCaR[0]) == tmp.size());
    for (int i = 1; i < tmp.size() - 1; i++)
      Vmrange_tauh[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
  }
#endif
	
}

ChannelCaR_GHK::~ChannelCaR_GHK() {}
