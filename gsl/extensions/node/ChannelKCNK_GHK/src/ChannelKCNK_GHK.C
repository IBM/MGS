#include "Lens.h"
#include "ChannelKCNK_GHK.h"
#include "CG_ChannelKCNK_GHK.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"
#include "NumberUtils.h"
#include <math.h>
#include <pthread.h>
#include <algorithm>

#define K_ION_FIXED 1
#define K_ION_DYNAMIC 2
#define CONC_K_IC  K_ION_FIXED

#if CONC_K_IC == K_ION_FIXED
#define K_i   (*(getSharedMembers().K_IC))
#else
//assuming index is 'i'
#define K_i   ((*K_IC)[i])
#endif
#define bo_bi   1   // ~ beta_o / beta_i  ~ partition coefficient 

void ChannelKCNK_GHK::update(RNG& rng) 
{  
   dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
    //dyn_var_t K_i = (*K_IC)[i];


#if CHANNEL_KCNK == KCNK_GHK_TUAN_2017
    {
      // NOTE: Some models use m_inf and tau_m to estimate m
//      dyn_var_t ma = AMC * vtrap(v-AMV, AMD);
//      dyn_var_t mb = BMC * exp((v-BMV) / BMD);
//      dyn_var_t tau_m = 1.0 / (ma + mb);
//      dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m * 2);
//
//      dyn_var_t tau_h = 70.0;  //[msec] - in NEURON code (due to Q10fact =3.0 at 22^C)
//      dyn_var_t qh = dt * getSharedMembers().Tadj / (tau_h * 2);
//
//      dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
//      dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));
//
//      m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
//      h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
//      // E_Ca[i] = (0.04343 * *(getSharedMembers().T) *
//      //           log(*(getSharedMembers().K_EC) / (*K_IC)[i]));
//      P_K[i] = P_Kbar[i] * m[i] * m[i] * (frac_inact * h[i] + (1.0 - frac_inact));
      ////dyn_var_t tmp = exp(-v * zKF_R / (*getSharedMembers().T));
      ////// NOTE: P_K [um/ms], Vm [mV], Cai/o [uM], F [C/mol] or [mJ/(mV.mol)]
      //////     R [mJ/(mol.K)]
      ////I_K[i] = P_K[i] * zK2F2_R / (*(getSharedMembers().T)) * v *
      ////          ((*K_IC)[i] - *(getSharedMembers().K_EC) * tmp) /
      ////          (1 - tmp);  // [pA/um^2]
      ////NOTE: Tuan added 0.314
      //dyn_var_t tmp = zKF_R * v / (*getSharedMembers().T); 
      ////I_K[i] = 1e-3 * P_K[i] * zK * zF * (-(K_i)* vtrap(-tmp, 1) - 0.314 * *(getSharedMembers().K_EC) * vtrap(tmp, 1));
      ////I_K[i] = 1e-3 * P_K[i] * zK * zF * 
      ////  (K_i * tmp + (K_i - 0.314 * *(getSharedMembers().K_EC)) * vtrap(tmp, 1));
      //I_K[i] = 1e-3 * P_K[i] * zK * zF * 
      //  (K_i * tmp + (K_i -  *(getSharedMembers().K_EC)) * vtrap(tmp, 1));
      I_K[i] = update_current(v, K_i, i);  // [pA/um^2]
#ifdef CONSIDER_DI_DV
      dyn_var_t I_K_dv = update_current(v+0.001, K_i, i);  // [pA/um^2]
      conductance_didv[i] = (I_K_dv - I_K[i])/(0.001);
#endif
    }
#endif
  }

}

void ChannelKCNK_GHK::initialize(RNG& rng) 
{
  assert(branchData);
  unsigned size = branchData->size;
  assert(V);
  assert(P_Kbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (m.size() != size) m.increaseSizeTo(size);
  if (h.size() != size) h.increaseSizeTo(size);
  if (P_K.size() != size) P_K.increaseSizeTo(size);
  if (I_K.size() != size) I_K.increaseSizeTo(size);
#ifdef CONSIDER_DI_DV
  if (conductance_didv.size() != size) conductance_didv.increaseSizeTo(size);
#endif
  // initialize
  dyn_var_t P_Kbar_default = P_Kbar[0];
  if (Pbar_dists.size() > 0 and Pbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either Pbar_dists or Pbar_branchorders on "
                 "GHK-formula KCNK channel "
                 "Channels Param" << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
    if (Pbar_dists.size() > 0)
    {
      unsigned int j;
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
      P_Kbar[i] = Pbar_values[j];
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
        P_Kbar[i] = Pbar_values[j - 1];
      }
      else if (j < Pbar_values.size())
        P_Kbar[i] = Pbar_values[j];
      else
        P_Kbar[i] = P_Kbar_default;
    }
    else
    {
      P_Kbar[i] = P_Kbar_default;
    }
  }
  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];
    //dyn_var_t K_i = (*K_IC)[i];
#ifdef CONSIDER_DI_DV
    conductance_didv[i] = 0.0;
#endif
  }
}

dyn_var_t ChannelKCNK_GHK::update_current(dyn_var_t v, dyn_var_t conc_K_i, int i)
{// voltage v (mV) and return current density I_K(pA/um^2)
  //NOTE: [K] is in unit of mM
    ////I_K[i] = 1e-3 * P_K[i] * zK * zF * 
    ////  (conc_K_i * tmp + (conc_K_i - beta_o/beta_i * *(getSharedMembers().K_EC)) * vtrap(tmp, 1));
    dyn_var_t tmp = zKF_R * v / (*getSharedMembers().T); 
#if 0
    dyn_var_t result = 1e-3 * P_K[i] * zK * zF * 
      (conc_K_i * tmp + (conc_K_i - bo_bi * *(getSharedMembers().K_EC)) * vtrap(tmp, 1.0));
#else
    // the equivalent form - Clay 2009
    // NOTE: If using this, we can explicitly pass E_K and need only to modify K_extracellular
    //  to study the effect of K_o when E_K is the same
    dyn_var_t tmp2 = zKF_R * (v - getSharedMembers().E_K[0]) / (*getSharedMembers().T); 
    dyn_var_t result = 1e-3 * P_K[i] * zK * zF * 
      *(getSharedMembers().K_EC) * (tmp2 - bo_bi) * vtrap(tmp, 1.0);
#endif
    return result;
}


ChannelKCNK_GHK::~ChannelKCNK_GHK() 
{
}

