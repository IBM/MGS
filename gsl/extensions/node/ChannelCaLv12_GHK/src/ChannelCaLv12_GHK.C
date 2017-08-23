#include "Lens.h"
#include "ChannelCaLv12_GHK.h"
#include "CG_ChannelCaLv12_GHK.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"
#include "NumberUtils.h"
#include "MaxComputeOrder.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

static pthread_once_t once_CaLv12_GHK = PTHREAD_ONCE_INIT;

// This is an implementation of L-type alpha1.2 Ca2+ channel
//              CaLv12_GHK current
//
#if CHANNEL_CaLv12 == CaLv12_GHK_Standen_Stanfield_1982_option1 || \
    CHANNEL_CaLv12 == CaLv12_GHK_Standen_Stanfield_1982_option2
//   Experimental data showed that Vm-dependent activation + inactivation
//   is not enough to fit the data
//    Ca2+-dependent activation + inactivation
//    Vm-dependent activation only (though data shows biphasic behavior)
//    ICa = PCabar * m^3* h * zCa^2 * F^2 * Vm/(RT) * 
//          ( gamma_o [Ca]_o - gamma_i [Ca]_i * exp(zCa * F * Vm / (RT)) ) /
//          (1- exp(zCa * F * Vm / (RT)))
//    with m(Vm,t) and h(Cai,t) 
//    gamma_i = gamma_o = 1
//    Pcabar = 6*10^-5 cm/sec
//NOTE: Later models use gamma_i = 1.0; gamma_o  = 0.341 (for Ca2+, Ba2+)
#define AMC -0.013
#define AMV -20
#define AMD -3
#define BMC 0.031
#define BMV -20
#define BMD -25
#define K_h 1.0 // [uM]
//#endif
#elif CHANNEL_CaLv12 == CaLv12_GHK_WOLF_2005
//  The model still assume Vm-dependent activation and inactivation
//  Inactivation reference from 
//     1. Bell - ... - Dolphin (2001) 
//            Title: biophys. properties, pharmacol., modulation 
//      of human, neuronal L-type (a1D, Cav1.3) - J. Neurophysiol. (Fig. 2, pg. 819)
//  Activation reference from 
//     1. Churchill et al. (1998)
//     2. Kasai et al. (1992) tau_m  (Fig. 15)
//
// same kinetics as that of CaLv13 of Wolf2005, just Vhalf-activated is higher
// minf(Vm) = 1/(1+exp((Vm-Vh)/k))
// hinf(Vm) = 1/(1+exp(Vm-Vh)/k)
//malpha = c * (v-vm) / ( exp((v-vm)/k) - 1  )
//mbeta = cpr * exp(v/kpr)    : Kasai 1992, fig 15
#define VHALF_M -8.9       // [mV]
#define k_M -6.7           // [mV]
#define VHALF_H -13.4
#define k_H 11.9
#define frac_inact 0.17
//#define AMC 0.1194 (used with 35^C)
#define AMC 0.0398      // [1/(ms.mV)]
#define AMV -8.124
#define AMD 9.005
//#define BMC 2.97  (used with 35^C)
#define BMC 0.99        // [1/ms]
#define BMV 0.0
#define BMD 31.4 
#else
#define frac_inact 1.0
NOT IMPLEMENTED YET
#endif

// NOTE: vtrap(x,y) = x/(exp(x/y)-1)
dyn_var_t ChannelCaLv12_GHK::vtrap(dyn_var_t x, dyn_var_t y)
{
  return (fabs(x / y) < SMALL ? y * (1 - x / y / 2) : x / (exp(x / y) - 1));
}

// GOAL: To meet second-order derivative, the gates is calculated to 
//     give the value at time (t0+dt/2) using data voltage v(t0)
//  NOTE: 
//    If steady-state formula is used, then the calculated value of gates
//            is at time (t0); but as steady-state, value at time (t0+dt/2) is the same
//    If non-steady-state formula (dy/dt = f(v)) is used, then 
//        once gate(t0) is calculated using v(t0)
//        we need to estimate gate(t0+dt/2)
//                  gate(t0+dt/2) = gate(t0) + f(v(t0)) * dt/2 
void ChannelCaLv12_GHK::initialize(RNG& rng)
{
  pthread_once(&once_CaLv12_GHK, initialize_others);
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
                 "GHK-formula Ca2+ Lv12 channel "
                 "Channels Param" << typeid(*this).name() << std::endl;
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
    dyn_var_t cai = (*Ca_IC)[i];
#if CHANNEL_CaLv12 == CaLv12_GHK_Standen_Stanfield_1982_option1
    {
      //NOTE: assume 1 binding site; binding is sufficiently fast (i.e. treated instantaneous)
      //   Ca + R <=>[K_h]  CaR   
      //   NOTE: R = channel, CaR = inactivated channel
      //   NOTE: K_h = alpha_h / beta_h
      h[i] = K_h / (cai + K_h);
      dyn_var_t am = AMC * vtrap((v - AMV), AMD);  // [1/ms]
      dyn_var_t bm = BMC * (exp((v - BMV) / BMD)); // [1/ms]
      m[i] = am / (am + bm);  // steady-state value
      PCa[i] = PCabar[i] * pow(m[i],3) *  h[i];
      I_Ca[i] = update_current(v, cai, i);  // [pA/um^2]
    }
#elif CHANNEL_CaLv12 == CaLv12_GHK_Standen_Stanfield_1982_option2
    {
      //NOTE: assume 1 binding site; binding is sufficiently fast (i.e. treated instantaneous)
      //   Ca + R <=>[K_h]  CaR   
      //   NOTE: R = channel, CaR = inactivated channel
      //   NOTE: K_h = alpha_h / beta_h
      //   dh/dt = alpha_h * (1-h) - beta_h * [Ca2+] * h;
      //h[i] = K_h / (cai + K_h);
      assert(0) ; //not completed
      dyn_var_t am = AMC * vtrap((v - AMV), AMD);  // [1/ms]
      dyn_var_t bm = BMC * (exp((v - BMV) / BMD)); // [1/ms]
      m[i] = am / (am + bm);  // steady-state value
      PCa[i] = PCabar[i] * pow(m[i],3) *  h[i];
      I_Ca[i] = update_current(v, cai, i); // [pA/um^2]
    }
#elif CHANNEL_CaLv12 == CaLv12_GHK_WOLF_2005
    {
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));  // steady-state values time (t0) and (t0+dt/2) are the same
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
    PCa[i] = PCabar[i] * m[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
    //dyn_var_t tmp = exp(-v * zCaF_R / (*getSharedMembers().T));
    //// NOTE: PCa [um/ms], Vm [mV], Cai/o [uM], F [C/mol] or [mJ/(mV.mol)]
    ////     R [mJ/(mol.K)]
    //I_Ca[i] = PCa[i] * (zCa2F2_R / (*(getSharedMembers().T))) * v *
    //          ((*Ca_IC)[i] - *(getSharedMembers().Ca_EC) * tmp) /
    //          (1 - tmp);  // [pA/um^2]
    //NOTE: Tuan added 0.314
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
void ChannelCaLv12_GHK::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
    dyn_var_t cai = (*Ca_IC)[i];
#if CHANNEL_CaLv12 == CaLv12_GHK_Standen_Stanfield_1982_option1
    {
      h[i] = K_h / (cai + K_h);
      dyn_var_t am = AMC * vtrap((v - AMV), AMD);  // [1/ms]
      dyn_var_t bm = BMC * (exp((v - BMV) / BMD)); // [1/ms]
      dyn_var_t m_inf = am / (am + bm);  // steady-state value
      dyn_var_t tau_m = 1.0 / (ma + mb);
      dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m * 2);
      // see Rempe-Chopp (2006)
      m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);

      PCa[i] = PCabar[i] * pow(m[i],3) *  h[i];
      I_Ca[i] = update_current(v, cai, i);  // [pA/um^2]

#ifdef CONSIDER_DI_DV
      dyn_var_t I_Ca_dv = update_current(v+0.001, cai, i);  // [pA/um^2]
      conductance_didv[i] = (I_Ca_dv - I_Ca[i])/(0.001);
#endif
    }
    assert(0);
#elif CHANNEL_CaLv12 == CaLv12_GHK_Standen_Stanfield_1982_option2
    assert(0);
#elif CHANNEL_CaLv12 == CaLv12_GHK_WOLF_2005
    {
      // NOTE: Some models use m_inf and tau_m to estimate m
      //dyn_var_t ma = 0.1194 * (v + 8.124) / (exp((v + 8.124) / 9.005) - 1); //these values at at 35^C
      dyn_var_t ma = AMC * vtrap(v-AMV, AMD);
      //dyn_var_t mb = 2.97 * exp((v) / 31.4); //these values are at 35^C
      dyn_var_t mb = BMC * exp((v-BMV) / BMD);
      dyn_var_t tau_m = 1.0 / (ma + mb);
      dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m * 2);

      //dyn_var_t tau_h = 14.77;  //[msec] - in paper (for 35^C)
      dyn_var_t tau_h = 44.3;  //[msec] - in NEURON code (due to Q10fact =3.0 at 22^C)
      dyn_var_t qh = dt * getSharedMembers().Tadj / (tau_h * 2);

      dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
      dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));

      // see Rempe-Chopp (2006)
      m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
      h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
      PCa[i] = PCabar[i] * m[i] * m[i] * (frac_inact * h[i] + (1.0 - frac_inact));

      //dyn_var_t tmp = exp(-v * zCaF_R / (*getSharedMembers().T));
      //// NOTE: PCa [um/ms], Vm [mV], Cai/o [uM], F [C/mol] or [mJ/(mV.mol)]
      ////     R [mJ/(mol.K)]
      //I_Ca[i] = PCa[i] * (zCa2F2_R / (*(getSharedMembers().T))) * v *
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

dyn_var_t ChannelCaLv12_GHK::update_current(dyn_var_t v, dyn_var_t cai, int i)
{// voltage v (mV) and return current density I_Ca(pA/um^2)
    ////I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * (-(cai)* vtrap(-tmp, 1) - 0.314 * 
    //                       *(getSharedMembers().Ca_EC) * vtrap(tmp, 1));
    ////I_Ca[i] = 1e-6 * PCa[i] * zCa * zF * 
    ////  (cai * tmp + (cai - 0.314 * *(getSharedMembers().Ca_EC)) * vtrap(tmp, 1));
    dyn_var_t tmp = zCaF_R * v / (*getSharedMembers().T); 
    dyn_var_t result = 1e-6 * PCa[i] * zCa * zF * 
      (cai * tmp + (cai -  *(getSharedMembers().Ca_EC)) * vtrap(tmp, 1));
    return result;
}

void ChannelCaLv12_GHK::initialize_others()
{
}


ChannelCaLv12_GHK::~ChannelCaLv12_GHK() {}
