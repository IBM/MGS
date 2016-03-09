#include "Lens.h"
#include "PumpSERCA.h"
#include "CG_PumpSERCA.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"
#include "NTSMacros.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

#if PUMP_SERCA == SERCA_Tran_Crampin_2009
  // NOTE: Tran - ... - Crampin (2009)  thermodynamic model 
	//    Here the two state model is used
#define Kp_nsr 2.24e3              // [uM]
#define Kp_myo 0.91e3              // [uM]
#define coupling_Ca 2  // number of Ca2+ translocated per molecule cycle

#elif PUMP_SERCA == SERCA_Klein_Schneider_1991
   // NOTE: Klein - .... - Schneider (1991) Michaelis-Menten-based formula 
	 //       derived for frog skeletal muscle 
	 //       with 4Ca+ binding in cytosol for the pump to activate
	 //       No [Ca]SR dependent
#define coupling_Ca   4   // 4-Ca2+ binding for the pump to do the job
#define K_serca 0.28  // [uM] 
#define v_max   0.208   // [uM/msec]

#elif PUMP_SERCA == _COMPONENT_UNDEFINED
  // do nothing
#else
  NOT IMPLEMENTED YET
#endif

void PumpSERCA::initialize(RNG& rng)
{
#ifdef DEBUG_ASSERT
  assert(branchData);
#endif
  unsigned size = branchData->size;
#ifdef DEBUG_ASSERT
  assert(Ca_ER);
  assert(Ca_ER->size() == size);
  assert(Ca_IC);
  assert(Ca_IC->size() == size);
  assert(SERCAConc);
  assert(SERCAConc->size() == size);
#endif
  // allocate
  if (J_Ca.size() != size) J_Ca.increaseSizeTo(size);
  // initialize
  dyn_var_t SERCAConc_default = SERCAConc[0];
  if (SERCAConc_dists.size() > 0 and SERCAConc_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either SERCAConc_dists or SERCAConc_branchorders on "
                 "GHK-formula Ca2+ channel "
                 "Channels Param" << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
    if (SERCAConc_dists.size() > 0)
    {
      int j;
      assert(SERCAConc_values.size() == SERCAConc_dists.size());
      for (j = 0; j < SERCAConc_dists.size(); ++j)
      {
        if ((*dimensions)[_cptindex]->dist2soma < SERCAConc_dists[j]) break;
      }
      if (j < SERCAConc_values.size())
        SERCAConc[i] = SERCAConc_values[j];
      else
        SERCAConc[i] = SERCAConc_default;
    }
    /*else if (SERCAConc_values.size() == 1) {
SERCAConc[i] = SERCAConc_values[0];
} */
    else if (SERCAConc_branchorders.size() > 0)
    {
      int j;
      assert(SERCAConc_values.size() == SERCAConc_branchorders.size());
      SegmentDescriptor segmentDescriptor;
      // const special_value = -1;
      for (j = 0; j < SERCAConc_branchorders.size(); ++j)
      {
        if (segmentDescriptor.getBranchOrder(branchData->key) ==
            SERCAConc_branchorders[j])
          break;
      }
      // if (j == SERCAConc_branchorders.size() and SERCAConc_branchorders[j-1] ==
      // special_value)
      if (j == SERCAConc_branchorders.size() and
          SERCAConc_branchorders[j - 1] == GlobalNTS::anybranch_at_end)
      {
        SERCAConc[i] = SERCAConc_values[j - 1];
      }
      else if (j < SERCAConc_values.size())
        SERCAConc[i] = SERCAConc_values[j];
      else
        SERCAConc[i] = SERCAConc_default;
    }
    else
    {
      SERCAConc[i] = SERCAConc_default;
    }
  }
	
  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t cai = (*Ca_IC)[i];   //[uM]
    dyn_var_t casr = (*Ca_ER)[i];  //[uM]

#if PUMP_SERCA == SERCA_Tran_Crampin_2009
    dyn_var_t TCa_ic2 = pow(cai / Kp_myo, 2);  // unitless
    dyn_var_t TCa_sr2 = pow(casr / Kp_nsr, 2);   // unitless
    dyn_var_t Dcycle = 0.104217 + 17.923 * TCa_sr2 +
                       TCa_ic2 * (1.75583e6 + 7.61673e6 * TCa_sr2) +
                       TCa_ic2 * (6.08463e11 + 4.50544e11 * TCa_sr2);

    dyn_var_t vcycle =
        (3.24873e12 * pow(TCa_ic2, 2) +
         TCa_ic2 * (9.17846e6 - 11478.2 * TCa_sr2) - 0.329904 * TCa_sr2) /
        Dcycle;                              // [1/msec]
    J_Ca[i] = - coupling_Ca * vcycle * SERCAConc[i];  // [uM/ms]
#elif PUMP_SERCA == SERCA_Klein_Schneider_1991 
		J_Ca[i] = - v_max * (pow(cai, coupling_Ca))/ (pow(cai, coupling_Ca) + pow(K_serca, coupling_Ca)); 
#endif
  }
}

void PumpSERCA::update(RNG& rng) {
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t cai = (*Ca_IC)[i];   //[uM]
    dyn_var_t casr = (*Ca_ER)[i];  //[uM]

#if PUMP_SERCA == SERCA_Tran_Crampin_2009
    dyn_var_t TCa_ic2 = pow(cai / Kp_myo, 2);  // unitless
    dyn_var_t TCa_sr2 = pow(casr / Kp_nsr, 2);   // unitless
    dyn_var_t Dcycle = 0.104217 + 17.923 * TCa_sr2 +
                       TCa_ic2 * (1.75583e6 + 7.61673e6 * TCa_sr2) +
                       TCa_ic2 * (6.08463e11 + 4.50544e11 * TCa_sr2);

    dyn_var_t vcycle =
        (3.24873e12 * pow(TCa_ic2, 2) +
         TCa_ic2 * (9.17846e6 - 11478.2 * TCa_sr2) - 0.329904 * TCa_sr2) /
        Dcycle;                              // [1/msec]
    J_Ca[i] = - coupling_Ca * vcycle * SERCAConc[i];  // [uM/ms]
#elif PUMP_SERCA == SERCA_Klein_Schneider_1991 
		J_Ca[i] = - v_max * (pow(cai, coupling_Ca))/ (pow(cai, coupling_Ca) + pow(K_serca, coupling_Ca)); 
#endif
  }
}

void PumpSERCA::setPointers(const String& CG_direction,
                              const String& CG_component,
                              NodeDescriptor* CG_node, Edge* CG_edge,
                              VariableDescriptor* CG_variable,
                              Constant* CG_constant,
                              CG_PumpSERCAInAttrPSet* CG_inAttrPset,
                              CG_PumpSERCAOutAttrPSet* CG_outAttrPset)
{
}

PumpSERCA::~PumpSERCA() {}
