// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "ChannelCaPQ_GHK.h"
#include "CG_ChannelCaPQ_GHK.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"
#include "NumberUtils.h"
#include "MaxComputeOrder.h"

#include <math.h>
#include <pthread.h>
#include <algorithm>

#define SMALL 1.0E-6

#if CHANNEL_CaPQ == CaPQ_GHK_WOLF_2005
#define bo_bi   1   // ~ beta_o / beta_i  ~ partition coefficient 
#elif CHANNEL_CaPQ == CaPQ_GHK_TUAN_2017
#define bo_bi   0.314   // ~ beta_o / beta_i  ~ partition coefficient 
#else
#define bo_bi   1   // ~ beta_o / beta_i  ~ partition coefficient 
#endif
static pthread_once_t once_CaPQ_GHK = PTHREAD_ONCE_INIT;

// This is an implementation of L-type alpha1.2 Ca2+ channel
//              CaPQ_GHK current
//
#if CHANNEL_CaPQ == CaPQ_GHK_WOLF_2005 || \
    CHANNEL_CaPQ == CaPQ_GHK_TUAN_2017
// same kinetics as that of CaLv12 of Wolf2005, just Vhalf-activated is lower
// ONLY activation
//  Activation reference from 
//     1. Churchill et al. (1998) for slope (Fig. 5)
//     2. Randal et al. (1995) tau_m  (Fig. 13)
//
// minf(Vm) = 1/(1+exp((Vm-Vh)/k))
// hinf(Vm) = 1/(1+exp(Vm-Vh)/k)
//#define VHALF_H -13.4
//#define k_H 11.9
//#define frac_inact  1
#define VHALF_M -9.0
#define k_M -6.6
#else
//#define frac_inact  1.0
NOT IMPLEMENTED YET
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
void ChannelCaPQ_GHK::initialize(RNG& rng) 
{
  pthread_once(&once_CaPQ_GHK, initialize_others);
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
    std::cerr << "ERROR: Use either Pbar_dists or Pbar_branchorders on "
                 "GHK-formula Ca2+ PQ-type channel "
                 "Channels Param" << typeid(*this).name() << std::endl;
    assert(0);
  }
  //initialize Pbar
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
  //calculate currents at time (t) and di_dv 
  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];
    dyn_var_t cai = (*Ca_IC)[i];
#if CHANNEL_CaPQ == CaPQ_GHK_WOLF_2005 || \
    CHANNEL_CaPQ == CaPQ_GHK_TUAN_2017
    {
      m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));  // steady-state values
      //h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
      PCa[i] = PCabar[i] * m[i] * m[i] ;
      ////dyn_var_t tmp = exp(-v * zCaF_R / (*getSharedMembers().T));
      ////// NOTE: PCa [um/ms], Vm [mV], Cai/o [uM], F [C/mol] or [mJ/(mV.mol)]
      //////     R [mJ/(mol.K)]
      ////I_Ca[i] = PCa[i] * zCa2F2_R / (*(getSharedMembers().T)) * v *
      ////          ((*Ca_IC)[i] - *(getSharedMembers().Ca_EC) * tmp) /
      ////          (1 - tmp);  // [pA/um^2]
      ////NOTE: Tuan added 0.314
      //dyn_var_t tmp = zCaF_R * v / (*getSharedMembers().T); 
      ////I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * (-(cai)* vtrap(-tmp, 1) - 0.314 * *(getSharedMembers().Ca_EC) * vtrap(tmp, 1));
      ////I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * 
      ////  (cai * tmp + (cai - 0.314 * *(getSharedMembers().Ca_EC)) * vtrap(tmp, 1));
      //I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * 
      //  (cai * tmp + (cai -  *(getSharedMembers().Ca_EC)) * vtrap(tmp, 1));

      I_Ca[i] = update_current(v, cai, i);  // [pA/um^2]
    }
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
void ChannelCaPQ_GHK::update(RNG& rng) 
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
    dyn_var_t cai = (*Ca_IC)[i];
#if CHANNEL_CaPQ == CaPQ_GHK_WOLF_2005 || \
    CHANNEL_CaPQ == CaPQ_GHK_TUAN_2017
    {
      // NOTE: Some models use m_inf and tau_m to estimate m
      //dyn_var_t tau_m = 0.377; //msec - in the paper (for 35^C)
      dyn_var_t tau_m = 1.13; //msec - in NEURON code (for 22^C)
      dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m * 2);
      dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));

      m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);

      PCa[i] = PCabar[i] * m[i] * m[i] ;
      ////dyn_var_t tmp = exp(-v * zCaF_R / (*getSharedMembers().T));
      ////// NOTE: PCa [um/ms], Vm [mV], Cai/o [uM], F [C/mol] or [mJ/(mV.mol)]
      //////     R [mJ/(mol.K)]
      ////I_Ca[i] = PCa[i] * zCa2F2_R / (*(getSharedMembers().T)) * v *
      ////          ((*Ca_IC)[i] - *(getSharedMembers().Ca_EC) * tmp) /
      ////          (1 - tmp);  // [pA/um^2]
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

dyn_var_t ChannelCaPQ_GHK::update_current(dyn_var_t v, dyn_var_t cai, int i)
{// voltage v (mV) and return current density I_Ca(pA/um^2)
    dyn_var_t tmp = zCaF_R * v / (*getSharedMembers().T); 
    dyn_var_t result = 1e-6 * PCa[i] * zCa * zF * 
      (cai * tmp + (cai - bo_bi * *(getSharedMembers().Ca_EC)) * vtrap(tmp, 1.0));
    return result;
}


void ChannelCaPQ_GHK::initialize_others()
{
}

ChannelCaPQ_GHK::~ChannelCaPQ_GHK() 
{
}

