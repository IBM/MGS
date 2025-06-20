// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "ChannelCaT_GHK.h"
#include "CG_ChannelCaT_GHK.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"
#include "NumberUtils.h"
#include "MaxComputeOrder.h"

#include <math.h>
#include <pthread.h>
#include <algorithm>

#define SMALL 1.0E-6

#if CHANNEL_CaT == CaT_GHK_WOLF_2005
#define bo_bi   1   // ~ beta_o / beta_i  ~ partition coefficient 
#elif CHANNEL_CaT == CaT_GHK_TUAN_2017
#define bo_bi   0.314   // ~ beta_o / beta_i  ~ partition coefficient 
#else
#define bo_bi   1   // ~ beta_o / beta_i  ~ partition coefficient 
#endif

static pthread_once_t once_CaT_GHK = PTHREAD_ONCE_INIT;

// This is an implementation of T-type Ca2+ channel
//              CaT_GHK current
//
#if CHANNEL_CaT == CaT_GHK_WOLF_2005 || \
    CHANNEL_CaT == CaT_GHK_TUAN_2017
//  Inactivation reference from
//     1. Chirchill et al. (1998) (Fig. 3) 
//  Activation reference from 
//     1. McRoy et al. (2001) (Fig. 7)
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
NOT IMPLEMENTED YET;
#endif


// GOAL: To meet second-order derivative, the gates is calculated to 
//     give the value at time (t0+dt/2) using data voltage v(t0)
//  NOTE: 
//    If steady-state formula is used, then the calculated value of gates
//            is at time (t0); but as steady-state, value at time (t0+dt/2) is the same
//    If non-steady-state formula (dy/dt = f(v)) is used, then 
//        once gate(t0) is calculated using v(t0)
//        we need to estimate gate(t0+dt/2)
//                  gate(t0+dt/2) = gate(t0) + f(v(t0)) * dt/2 
//     Current is calculated v(t0) and gates(t0)
void ChannelCaT_GHK::initialize(RNG& rng)
{
  pthread_once(&once_CaT_GHK, initialize_others);
  assert(branchData);
  unsigned size = branchData->size;
  assert(V);
  assert(PCabar.size() == size);
  assert(V->size() == size);

  // allocate
  if (m.size() != size) m.increaseSizeTo(size);
  if (h.size() != size) h.increaseSizeTo(size);
  if (PCa.size() != size) PCa.increaseSizeTo(size);
  if (I_Ca.size() != size) I_Ca.increaseSizeTo(size);
#ifdef CONSIDER_DI_DV
  if (conductance_didv.size() != size) conductance_didv.increaseSizeTo(size);
#endif
  // initialize
  dyn_var_t PCabar_default = PCabar[0];
  if (Pbar_dists.size() > 0 and Pbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either Pbar_dists or Pbar_branchorders on GHK-formula Ca2+ T-type channel "
                 "Channels Param for " << typeid(*this).name() << std::endl;
    assert(0);
  }
  //initialize
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
  //calculate at current at time t0
  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];
#ifdef MICRODOMAIN_CALCIUM
    dyn_var_t cai = (*Ca_IC)[i+_offset]; // [uM]
#else
    dyn_var_t cai = (*Ca_IC)[i];
#endif

#if CHANNEL_CaT == CaT_GHK_WOLF_2005 || \
    CHANNEL_CaT == CaT_GHK_TUAN_2017
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));  // steady-state values
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
    PCa[i] = PCabar[i] * m[i] * m[i] * m[i] * h[i];
    ////dyn_var_t tmp = exp(-v * zCaF_R / (*getSharedMembers().T));
    //////NOTE: PCa [um/ms], Vm [mV], Cai/o [uM], F [C/mol] or [mJ/(mV.mol)]
    //////     R [mJ/(mol.K)]
    ////I_Ca[i] = PCa[i] * zCa2F2_R / (*(getSharedMembers().T)) * 
    ////  v * ((*Ca_IC)[i] - *(getSharedMembers().Ca_EC) * tmp)/
    ////  (1- tmp); // [pA/um^2]
    ////NOTE: Tuan added 0.314
    //dyn_var_t tmp = zCaF_R * v / (*getSharedMembers().T); 
    ////I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * (-(cai)* vtrap(-tmp, 1) - 0.314 * *(getSharedMembers().Ca_EC) * vtrap(tmp, 1));
    ////I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * 
    ////  (cai * tmp + (cai - 0.314 * *(getSharedMembers().Ca_EC)) * vtrap(tmp, 1));
    //I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * 
    //  (cai * tmp + (cai -  *(getSharedMembers().Ca_EC)) * vtrap(tmp, 1));
    I_Ca[i] = update_current(v, cai, i);  // [pA/um^2]
#else
    NOT IMPLEMENTED YET;
#endif
#ifdef CONSIDER_DI_DV
    conductance_didv[i] = 0.0;
#endif

  }
}

// GOAL: update gates using v(t+dt/2) and gate(t-dt/2)
//   --> output gate(t+dt/2+dt)
//   of second-order accuracy at time (t+dt/2+dt) using trapezoidal rule
void ChannelCaT_GHK::update(RNG& rng)
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

#if CHANNEL_CaT == CaT_GHK_WOLF_2005 || \
    CHANNEL_CaT == CaT_GHK_TUAN_2017
    {
      // NOTE: Some models use m_inf and tau_m to estimate m
      // tau_m in the lookup table
      std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
      int index = low - Vmrange_taum.begin();
      //-->tau_m[i] = taumCaT[index];
      // NOTE: dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m[i] * 2);
      //dyn_var_t qm = dt * getSharedMembers().Tadj / (taumCaT[index] * 2);
      dyn_var_t taum;
      if (index == 0)
        taum = taumCaT[0];
      else
        taum = linear_interp(Vmrange_taum[index-1], taumCaT[index-1], 
            Vmrange_taum[index], taumCaT[index], v);
      dyn_var_t qm = dt * getSharedMembers().Tadj / (taum * 2);
      /* no need to search as they both use the same Vmrange
       * IF NOT< make sure you add this code
       std::vector<dyn_var_t>::iterator low= std::lower_bound(Vmrange_tauh.begin(),
       Vmrange_tauh.end(), v);
       int index = low-Vmrange_tauh.begin();
       */
      //dyn_var_t qh = dt * getSharedMembers().Tadj / (tauhCaT[index] * 2);
      dyn_var_t tauh;
      if (index == 0)
        tauh = tauhCaT[0];
      else
        tauh = linear_interp(Vmrange_tauh[index-1], tauhCaT[index-1], 
            Vmrange_tauh[index], tauhCaT[index], v);
      dyn_var_t qh = dt * getSharedMembers().Tadj / (tauh * 2);

      dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
      dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));

      m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
      h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
      //E_Ca[i] = (0.04343 * *(getSharedMembers().T) *
      //           log(*(getSharedMembers().Ca_EC) / (*Ca_IC)[i]));
      PCa[i] = PCabar[i] * m[i] * m[i] * m[i] * h[i];
      ////dyn_var_t tmp = exp(-v * zCaF_R / (*getSharedMembers().T));
      //////NOTE: PCa [um/ms], Vm [mV], Cai/o [uM], F [C/mol] or [mJ/(mV.mol)]
      //////     R [mJ/(mol.K)]
      ////I_Ca[i] = PCa[i] * zCa2F2_R / (*(getSharedMembers().T)) * 
      ////  v * ((*Ca_IC)[i] - *(getSharedMembers().Ca_EC) * tmp)/
      ////  (1- tmp); // [pA/um^2]
      ////NOTE: Tuan added 0.314
      //dyn_var_t tmp = zCaF_R * v / (*getSharedMembers().T); 
      ////I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * (-(cai)* vtrap(-tmp, 1) - 0.314 * *(getSharedMembers().Ca_EC) * vtrap(tmp, 1));
      ////I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * 
      ////  (cai * tmp + (cai - 0.314 * *(getSharedMembers().Ca_EC)) * vtrap(tmp, 1));
      //I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * 
      //  (cai * tmp + (cai -  *(getSharedMembers().Ca_EC)) * vtrap(tmp, 1));
      I_Ca[i] = update_current(v, cai, i);  // [pA/um^2]

#ifdef CONSIDER_DI_DV
      dyn_var_t I_Ca_dv = update_current(v+0.001, cai, i);  // [pA/um^2]
      conductance_didv[i] = (I_Ca_dv - I_Ca[i])/(0.001);
#endif

    }
#endif
    /*
     * TUAN TODO: think about stochastic modelling
     * I_Ca[i] = Nopen * P_Ca_singlechannel * ...
     * with Nopen is from 0 to ... Nchannelpercompartment
     * Nchannelpercompartment = PCa*surfacearea_compartment/P_Ca_singlechannel
     * And use the Markov-based model for a single channel to determine
     * Nopen
    I_Ca[i] = PCa[i] * zCa2F2_R / (*(getSharedMembers().T)) * v *
              ((*Ca_IC)[i] - *(getSharedMembers().Ca_EC) * tmp) /
              (1 - tmp);  // [pA/um^2]
    */
  }
}

dyn_var_t ChannelCaT_GHK::update_current(dyn_var_t v, dyn_var_t cai, int i)
{// voltage v (mV) and return current density I_Ca(pA/um^2)
    ////I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * (-(cai)* vtrap(-tmp, 1) - 0.314 * 
    //                       *(getSharedMembers().Ca_EC) * vtrap(tmp, 1));
    ////I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * 
    ////  (cai * tmp + (cai - 0.314 * *(getSharedMembers().Ca_EC)) * vtrap(tmp, 1));
    dyn_var_t tmp = zCaF_R * v / (*getSharedMembers().T); 
    dyn_var_t result = 1e-6 * PCa[i] * zCa * zF * 
      (cai * tmp + (cai -  *(getSharedMembers().Ca_EC)) * vtrap(tmp, 1.0));
    return result;
}

void ChannelCaT_GHK::initialize_others()
{
#if CHANNEL_CaT == CaT_GHK_WOLF_2005 || \
    CHANNEL_CaT == CaT_GHK_TUAN_2017
  {
    std::vector<dyn_var_t> tmp(_Vmrange_taum,
                               _Vmrange_taum + LOOKUP_TAUM_LENGTH);
    assert((sizeof(taumCaT) / sizeof(taumCaT[0])) == tmp.size());
    //Vmrange_taum.resize(tmp.size()-2);
    //for (unsigned long i = 1; i < tmp.size() - 1; i++)
    //  Vmrange_taum[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
    Vmrange_taum = tmp;
  }
  {
    std::vector<dyn_var_t> tmp(_Vmrange_tauh,
                               _Vmrange_tauh + LOOKUP_TAUH_LENGTH);
    assert(sizeof(tauhCaT) / sizeof(tauhCaT[0]) == tmp.size());
    //Vmrange_tauh.resize(tmp.size()-2);
    //for (unsigned long i = 1; i < tmp.size() - 1; i++)
    //  Vmrange_tauh[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
    Vmrange_tauh = tmp;
  }
#endif
}

ChannelCaT_GHK::~ChannelCaT_GHK() {}
#ifdef MICRODOMAIN_CALCIUM
void ChannelCaT_GHK::setCalciumMicrodomain(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ChannelCaT_GHKInAttrPSet* CG_inAttrPset, CG_ChannelCaT_GHKOutAttrPSet* CG_outAttrPset) 
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
