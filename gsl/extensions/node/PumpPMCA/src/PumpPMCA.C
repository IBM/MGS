#include "Lens.h"
#include "PumpPMCA.h"
#include "CG_PumpPMCA.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"
#include "NTSMacros.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>


#if PUMP_PMCA == PMCA_PUMPRATE_CONSTANT
    // PMCA_Traub_Llinas_1997
		// NOTE : a lumped mechanism with neutral effect on Vm
		//        and constant rate
#elif PUMP_PMCA == PMCA_PUMPRATE_CONSTANT_DYNAMICS
    // Enable the assignment of different rate on different branches
	//  via the ChanParam.par file
#elif PUMP_PMCA == PMCA_PUMPRATE_VOLTAGE_FUNCTION
    // PMCA_Zador_Koch_Brown_1990
		// NOTE : a lumped mechanism with neutral effect on Vm
		//        and rate as a function of Vm
  #define tc_factor 17.7   // [msec] = time constant factor
  #define kV    35.0 // [mV] = steepness of Vm-dependency
#elif PUMP_PMCA == PMCA_Jafri_Rice_Winslow_1998
  // NOTE Jafri-Rice-Winslow (1998) Cardiac Ca2+ dynamics - role of RyR and SR load
	//      (Biophys.J.)
	// neutral PMCA with 1Ca2+ out and 2H+ in for 1 ATP molecule
  #define Km_pmca 0.5  // [uM] - dissociation constant of half-saturation

#elif PUMP_PMCA == PMCA_Greenstein_Winslow_2002
  // NOTE Greenstein-Winslow (2002) Integrative model of cardiac ventricular myocyte
	//      ... (Biophys.J.)
	// PMCA with 2Ca2+ out and 2H+ in for 1 ATP molecule
  #define Km_pmca 0.5  // [uM]
  #define eta_pmca 2
#endif

void PumpPMCA::initialize(RNG& rng)
{
//  pthread_once(&once_Pump_PMCA, initialize_others);
#ifdef DEBUG_ASSERT
  assert(branchData);
#endif
  unsigned size = branchData->size;
#ifdef DEBUG_ASSERT
  assert(V);
  assert(V->size() == size);
#endif
// allocate
#if PUMP_PMCA == PMCA_PUMPRATE_CONSTANT || \
    PUMP_PMCA == PMCA_PUMPRATE_CONSTANT_DYNAMICS || \
    PUMP_PMCA == PMCA_PUMPRATE_VOLTAGE_FUNCTION
  if (J_Ca.size() != size) J_Ca.increaseSizeTo(size);
 #if PUMP_PMCA == PMCA_PUMPRATE_CONSTANT_DYNAMICS 
  assert(tau.size() == size);
 #endif
 #if PUMP_PMCA == PMCA_PUMPRATE_VOLTAGE_FUNCTION 
  if (tau.size() != size) tau.increaseSizeTo(size);
 #endif
#else
  assert(IPMCAbar.size() == size);
  if (I_PMCA.size() != size) I_PMCA.increaseSizeTo(size);
  if (I_Ca.size() != size) I_Ca.increaseSizeTo(size);
  // initialize
  dyn_var_t IPMCAbar_default = IPMCAbar[0];
  if (IPMCAbar_dists.size() > 0 and IPMCAbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either IPMCAbar_dists or IPMCAbar_branchorders on "
                 "GHK-formula Ca2+ channel "
                 "Channels Param" << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
    if (IPMCAbar_dists.size() > 0)
    {
      unsigned long j;
      assert(IPMCAbar_values.size() == IPMCAbar_dists.size());
      for (j = 0; j < IPMCAbar_dists.size(); ++j)
      {
        if ((*dimensions)[i]->dist2soma < IPMCAbar_dists[j]) break;
      }
      if (j < IPMCAbar_values.size())
        IPMCAbar[i] = IPMCAbar_values[j];
      else
        IPMCAbar[i] = IPMCAbar_default;
    }
    /*else if (IPMCAbar_values.size() == 1) {
IPMCAbar[i] = IPMCAbar_values[0];
} */
    else if (IPMCAbar_branchorders.size() > 0)
    {
      unsigned long j;
      assert(IPMCAbar_values.size() == IPMCAbar_branchorders.size());
      SegmentDescriptor segmentDescriptor;
      // const special_value = -1;
      for (j = 0; j < IPMCAbar_branchorders.size(); ++j)
      {
        if (segmentDescriptor.getBranchOrder(branchData->key) ==
            IPMCAbar_branchorders[j])
          break;
      }
      // if (j == IPMCAbar_branchorders.size() and IPMCAbar_branchorders[j-1] ==
      // special_value)
      if (j == IPMCAbar_branchorders.size() and
          IPMCAbar_branchorders[j - 1] == GlobalNTS::anybranch_at_end)
      {
        IPMCAbar[i] = IPMCAbar_values[j - 1];
      }
      else if (j < IPMCAbar_values.size())
        IPMCAbar[i] = IPMCAbar_values[j];
      else
        IPMCAbar[i] = IPMCAbar_default;
    }
    else
    {
      IPMCAbar[i] = IPMCAbar_default;
    }
  }
#endif

  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];  // mV
// IMPORTANT: positive flux helps increase dCa/dt
#if PUMP_PMCA == PMCA_PUMPRATE_CONSTANT
    // PMCA_Traub_Llinas_1997
    dyn_var_t cai = (*Ca_IC)[i];  //[uM]
    J_Ca[i] = 1.0 / ((getSharedMembers().tau_pump)) *
              (getSharedMembers().Ca_equil - cai);  // [uM/ms]
#elif PUMP_PMCA == PMCA_PUMPRATE_CONSTANT_DYNAMICS
    // PMCA_Traub_Llinas_1997
    dyn_var_t cai = (*Ca_IC)[i];  //[uM]
    J_Ca[i] = 1.0 / ((tau[i])) *
              (getSharedMembers().Ca_equil - cai);  // [uM/ms]
#elif PUMP_PMCA == PMCA_PUMPRATE_VOLTAGE_FUNCTION
    // PMCA_Zador_Koch_Brown_1990
    dyn_var_t cai = (*Ca_IC)[i];       //[uM]
    tau[i] = tc_factor * exp(v / kV);  // [msec]
    J_Ca[i] = 1.0 / (tau[i]) * (getSharedMembers().Ca_equil - cai);  // [uM/ms]

#elif PUMP_PMCA == PMCA_Jafri_Rice_Winslow_1998
    dyn_var_t cai = (*Ca_IC)[i];  //[uM]
    // The saturation is incorporated in the denominator of the formula
    I_Ca[i] = IPMCAbar[i] * (cai) / (Km_pmca + cai); // [pA/um^2]
		I_PMCA[i] = 0.0;  // 1Ca2+ out - 2H+ in --> neutral

#elif PUMP_PMCA == PMCA_Greenstein_Winslow_2002
    dyn_var_t cai = (*Ca_IC)[i];  //[uM]
    // The saturation is incorporated in the denominator of the formula
    I_Ca[i] = IPMCAbar[i] * (pow(cai, eta_pmca)) /
              (pow(Km_pmca, eta_pmca) + pow(cai, eta_pmca));
    I_PMCA[i] = I_Ca[i]; // 2Ca2+ out - 2H+ in --> 2charges out
#elif PUMP_PMCA == _COMPONENT_UNDEFINED
// do nothing
#else
    NOT IMPLEMENTED YET
#endif
  }
}

void PumpPMCA::update(RNG& rng)
{
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];  // mV
// IMPORTANT: positive flux helps increase dCa/dt
#if PUMP_PMCA == PMCA_PUMPRATE_CONSTANT
    // PMCA_Traub_Llinas_1997
    dyn_var_t cai = (*Ca_IC)[i];  //[uM]
    J_Ca[i] = 1.0 / (*(getSharedMembers().tau_pump)) *
              (getSharedMembers().Ca_equil - cai);  // [uM/ms]
#elif PUMP_PMCA == PMCA_PUMPRATE_CONSTANT_DYNAMICS
    // PMCA_Traub_Llinas_1997
    dyn_var_t cai = (*Ca_IC)[i];  //[uM]
    J_Ca[i] = 1.0 / ((tau[i])) *
              (getSharedMembers().Ca_equil - cai);  // [uM/ms]
#elif PUMP_PMCA == PMCA_PUMPRATE_VOLTAGE_FUNCTION
    // PMCA_Zador_Koch_Brown_1990
    dyn_var_t cai = (*Ca_IC)[i];       //[uM]

    tau[i] = tc_factor * exp(v / kV);  // [msec]
    J_Ca[i] = 1.0 / (tau[i]) * (getSharedMembers().Ca_equil - cai);  // [uM/ms]

#elif PUMP_PMCA == PMCA_Jafri_Rice_Winslow_1998
    dyn_var_t cai = (*Ca_IC)[i];  //[uM]
    // The saturation is incorporated in the denominator of the formula
    I_Ca[i] = IPMCAbar[i] * (cai) / (Km_pmca + cai);

#elif PUMP_PMCA == PMCA_Greenstein_Winslow_2002
    dyn_var_t cai = (*Ca_IC)[i];  //[uM]
    // The saturation is incorporated in the denominator of the formula
    I_Ca[i] = IPMCAbar[i] * pow(cai, eta_pmca) /
              (pow(Km_pmca, eta_pmca) + pow(cai, eta_pmca));
    I_PMCA[i] = I_Ca[i]; // 2Ca2+ out - 2H+ in --> 2charges out
#elif PUMP_PMCA == _COMPONENT_UNDEFINED
// do nothing
#else
    NOT IMPLEMENTED YET
#endif
  }
}


PumpPMCA::~PumpPMCA() {}
