#include "Lens.h"
#include "ChannelCaLv12_GHK.h"
#include "CG_ChannelCaLv12_GHK.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>
static pthread_once_t once_CaLv12_GHK = PTHREAD_ONCE_INIT;

// This is an implementation of L-type alpha1.2 Ca2+ channel
//              CaLv12_GHK current
//
#if CHANNEL_CaLv12 == CaLv12_GHK_Standen_Stanfield_1982
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
//#endif
#elif CHANNEL_CaLv12 == CaLv12_GHK_WOLF_2005
//  The model still assume Vm-dependent activation and inactivation
//
// same kinetics as that of CaLv13 of Wolf2005, just Vhalf-activated is higher
// minf(Vm) = 1/(1+exp((Vm-Vh)/k))
// hinf(Vm) = 1/(1+exp(Vm-Vh)/k)
#define VHALF_M -8.9
#define k_M -6.7
#define VHALF_H -13.4
#define k_H 11.9
#define frac_inact 0.17
#else
#define frac_inact 1.0
NOT IMPLEMENTED YET
#endif

dyn_var_t ChannelCaLv12_GHK::vtrap(dyn_var_t x, dyn_var_t y)
{
  return (fabs(x / y) < SMALL ? y * (1 - x / y / 2) : x / (exp(x / y) - 1));
}

void ChannelCaLv12_GHK::initialize(RNG& rng)
{
  pthread_once(&once_CaLv12_GHK, initialize_others);
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
#if CHANNEL_CaLv12 == CaLv12_GHK_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));  // steady-state values
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
    PCa[i] = PCabar[i] * m[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
    dyn_var_t tmp = exp(-v * zCaF_R / (*getSharedMembers().T));
    // NOTE: PCa [um/ms], Vm [mV], Cai/o [uM], F [C/mol] or [mJ/(mV.mol)]
    //     R [mJ/(mol.K)]
    I_Ca[i] = PCa[i] * (zCa2F2_R / (*(getSharedMembers().T))) * v *
              ((*Ca_IC)[i] - *(getSharedMembers().Ca_EC) * tmp) /
              (1 - tmp);  // [pA/um^2]
#else
    NOT IMPLEMENTED YET
#endif
  }
}

void ChannelCaLv12_GHK::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_CaLv12 == CaLv12_WOLF_2005
    // NOTE: Some models use m_inf and tau_m to estimate m
    dyn_var_t ma = 0.1194 * (v + 8.124) / (exp((v + 8.124) / 9.005) - 1);
    dyn_var_t mb = 2.97 * exp((v) / 31.4);
    dyn_var_t tau_m = 1.0 / (ma + mb);
    dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m * 2);

    dyn_var_t tau_h = 14.77;  //[msec]
    dyn_var_t qh = dt * getSharedMembers().Tadj / (tau_h * 2);

    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
    // E_Ca[i] = (0.04343 * *(getSharedMembers().T) *
    //           log(*(getSharedMembers().Ca_EC) / (*Ca_IC)[i]));
    PCa[i] = PCabar[i] * m[i] * m[i] * (frac_inact * h[i] + (1.0 - frac_inact));

    dyn_var_t tmp = exp(-v * zCaF_R / (*getSharedMembers().T));
    // NOTE: PCa [um/ms], Vm [mV], Cai/o [uM], F [C/mol] or [mJ/(mV.mol)]
    //     R [mJ/(mol.K)]
    I_Ca[i] = PCa[i] * (zCa2F2_R / (*(getSharedMembers().T))) * v *
              ((*Ca_IC)[i] - *(getSharedMembers().Ca_EC) * tmp) /
              (1 - tmp);  // [pA/um^2]
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

void ChannelCaLv12_GHK::initialize_others()
{
}

void ChannelCaLv12_GHK::setPointers(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_ChannelCaLv12_GHKInAttrPSet* CG_inAttrPset,
    CG_ChannelCaLv12_GHKOutAttrPSet* CG_outAttrPset)
{
  _cptindex = CG_inAttrPset->idx;
}

ChannelCaLv12_GHK::~ChannelCaLv12_GHK() {}
