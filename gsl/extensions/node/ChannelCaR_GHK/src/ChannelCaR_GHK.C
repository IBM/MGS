#include "Lens.h"
#include "ChannelCaR_GHK.h"
#include "CG_ChannelCaR_GHK.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"
#include "NumberUtils.h"

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
//  Inactivation reference from 
//     1. Foehring ... Surmeier (2000) - Unique properties R-type Ca2+ currents in neocortical and neostriatal neurons - J. Neurophysiol. Fig. 7C
//  Activation reference from 
//     1. Churchill et al. (1998) for slope (Fig. 7)
//     2. Foehring et al. (2000) tau_m  (pg. 2230)
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

// NOTE: vtrap(x,y) = x/(exp(x/y)-1)
dyn_var_t ChannelCaR_GHK::vtrap(dyn_var_t x, dyn_var_t y)
{
  return (fabs(x / y) < SMALL ? y * (1 - x / y / 2) : x / (exp(x / y) - 1));
}

void ChannelCaR_GHK::initialize(RNG& rng)
{
  pthread_once(&once_CaR_GHK, initialize_others);
  assert(branchData);
  unsigned size = branchData->size;
  if (not V)
  {
    std::cerr << typeid(*this).name() << " needs Voltage as input in ChanParam\n";
    assert(V);
  }
  if (not Ca_IC)
  {
    std::cerr << typeid(*this).name() << " needs Calcium as input in ChanParam\n";
    assert(Ca_IC);
  }
  assert(PCabar.size() == size);
  assert(V->size() == size);
  // allocate
  if (m.size() != size) m.increaseSizeTo(size);
  if (h.size() != size) h.increaseSizeTo(size);
  if (PCa.size() != size) PCa.increaseSizeTo(size);
  if (I_Ca.size() != size) I_Ca.increaseSizeTo(size);
  // initialize
  dyn_var_t PCabar_default = PCabar[0];
  if (Pbar_dists.size() > 0 and Pbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either Pbar_dists or Pbar_branchorders on "
                 "GHK-formula Ca2+ R-type channel "
                 "Channels Param for " << typeid(*this).name() << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
    if (Pbar_dists.size() > 0)
    {
      unsigned int j;
      //NOTE: 'n' bins are splitted by (n-1) points
      if (Pbar_values.size() - 1 != Pbar_dists.size())
      {
        std::cerr << "Pbar_values.size = " << Pbar_values.size() 
          << "; Pbar_dists.size = " << Pbar_dists.size() << std::endl; 
      }
      assert(Pbar_values.size() -1 == Pbar_dists.size());
      for (j = 0; j < Pbar_dists.size(); ++j)
      {
        if ((*dimensions)[i]->dist2soma < Pbar_dists[j]) break;
      }
      PCabar[i] = Pbar_values[j];
    }
    else if (Pbar_branchorders.size() > 0)
    {
      unsigned int j;
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
#ifdef MICRODOMAIN_CALCIUM
    dyn_var_t cai = (*Ca_IC)[i+_offset]; // [uM]
#else
    dyn_var_t cai = (*Ca_IC)[i];
#endif

#if CHANNEL_CaR == CaR_GHK_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));  // steady-state values
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
    PCa[i] =
        PCabar[i] * m[i] * m[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
    //dyn_var_t tmp = exp(-v * zCaF_R / (*getSharedMembers().T));
    //// NOTE: PCa [um/ms], Vm [mV], Cai/o [uM], F [C/mol] or [mJ/(mV.mol)]
    ////     R [mJ/(mol.K)]
    //I_Ca[i] = PCa[i] * zCa2F2_R / (*(getSharedMembers().T)) * v *
    //          ((*Ca_IC)[i] - *(getSharedMembers().Ca_EC) * tmp) /
    //          (1 - tmp);  // [pA/um^2]
    //NOTE: Tuan added 0.314
    dyn_var_t tmp = zCaF_R * v / (*getSharedMembers().T); 
    //I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * (-(cai)* vtrap(-tmp, 1) - 0.314 * *(getSharedMembers().Ca_EC) * vtrap(tmp, 1));
    //I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * 
    //  (cai * tmp + (cai - 0.314 * *(getSharedMembers().Ca_EC)) * vtrap(tmp, 1));
    I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * 
      (cai * tmp + (cai -  *(getSharedMembers().Ca_EC)) * vtrap(tmp, 1));
#ifdef CONSIDER_DI_DV
    tmp = zCaF_R * (v+0.001) / (*getSharedMembers().T); 
    dyn_var_t I_Ca_dv = 1e-6 * PCa[i] * zCa * zF * 
      (cai * tmp + (cai -  *(getSharedMembers().Ca_EC)) * vtrap(tmp, 1));  // [pA/um^2]
    conductance_didv[i] = (I_Ca_dv - I_Ca[i])/(0.001);
#endif
#else
    NOT IMPLEMENTED YET;
#endif
  }
}

void ChannelCaR_GHK::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#ifdef MICRODOMAIN_CALCIUM
    dyn_var_t cai = (*Ca_IC)[i+_offset]; // [uM]
#else
    dyn_var_t cai = (*Ca_IC)[i];
#endif

#if CHANNEL_CaR == CaR_GHK_WOLF_2005
    // NOTE: Some models use m_inf and tau_m to estimate m
    //dyn_var_t tau_m = 1.6;  // msec - in paper (which assume 35^C)
    dyn_var_t tau_m = 5.1;  // msec - in NEURON code (due to Q10 =3.0 at 22^C)
    dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m * 2);

    std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_tauh.begin(), Vmrange_tauh.end(), v);
    int index = low - Vmrange_tauh.begin();
    //dyn_var_t qh = dt * getSharedMembers().Tadj / (tauhCaR[index] * 2);
    dyn_var_t tauh;
    if (index == 0)
      tauh = tauhCaR[0];
    else
      tauh = linear_interp(Vmrange_tauh[index-1], tauhCaR[index-1], 
        Vmrange_tauh[index], tauhCaR[index], v);
    dyn_var_t qh = dt * getSharedMembers().Tadj / (tauh * 2);

    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
    PCa[i] =
        PCabar[i] * m[i] * m[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
    //dyn_var_t tmp = exp(-v * zCaF_R / (*getSharedMembers().T));
    //// NOTE: PCa [um/ms], Vm [mV], Cai/o [uM], F [C/mol] or [mJ/(mV.mol)]
    ////     R [mJ/(mol.K)]
    //I_Ca[i] = PCa[i] * zCa2F2_R / (*(getSharedMembers().T)) * v *
    //          ((*Ca_IC)[i] - *(getSharedMembers().Ca_EC) * tmp) /
    //          (1 - tmp);  // [pA/um^2]
    //NOTE: Tuan added 0.314
    dyn_var_t tmp = zCaF_R * v / (*getSharedMembers().T); 
    //I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * (-(cai)* vtrap(-tmp, 1) - 0.314 * *(getSharedMembers().Ca_EC) * vtrap(tmp, 1));
    //I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * 
    //  (cai * tmp + (cai - 0.314 * *(getSharedMembers().Ca_EC)) * vtrap(tmp, 1));
    I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * 
      (cai * tmp + (cai -  *(getSharedMembers().Ca_EC)) * vtrap(tmp, 1));
#endif
  }
}


void ChannelCaR_GHK::initialize_others() 
{
#if CHANNEL_CaR == CaR_GHK_WOLF_2005
  {
    std::vector<dyn_var_t> tmp(_Vmrange_tauh,
                               _Vmrange_tauh + LOOKUP_TAUH_LENGTH);
    assert(sizeof(tauhCaR) / sizeof(tauhCaR[0]) == tmp.size());
    //Vmrange_tauh.resize(tmp.size()-2);
    //for (int i = 1; i < tmp.size() - 1; i++)
    //  Vmrange_tauh[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
    Vmrange_tauh = tmp;
  }
#endif
  
}

ChannelCaR_GHK::~ChannelCaR_GHK() {}

#ifdef MICRODOMAIN_CALCIUM
void ChannelCaR_GHK::setCalciumMicrodomain(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ChannelCaR_GHKInAttrPSet* CG_inAttrPset, CG_ChannelCaR_GHKOutAttrPSet* CG_outAttrPset) 
{
  microdomainName = CG_inAttrPset->domainName;
  int idxFound = 0;
  while((*(getSharedMembers().tmp_microdomainNames))[idxFound] != microdomainName)
  {
    idxFound++;
  }
  _offset = idxFound * branchData->size;
}
#endif
