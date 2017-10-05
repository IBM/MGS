#include "Lens.h"
#include "ExchangerNCX.h"
#include "CG_ExchangerNCX.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"
#include "MaxComputeOrder.h"
#include "NTSMacros.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

#define NCX_KIMURA_BASED_FORMULA   1
#define NCX_WEBER_BASED_FORMULA    2

#if EXCHANGER_NCX ==  NCX_Kimura_Miyamae_Noma_1987 
  //originally developed for cardiac muscle
  // There is no saturation in this model
  // However, in experimental data, when Vm is far from E_NaCa; then the I-V curve saturates
  //    --> check Weber et al. for saturation model
#define NCX_FORMULA_BASED   NCX_KIMURA_BASED_FORMULA
  // NCX
  // Iforward = [Na]_i^3 * Ca_o * exp( (n-2) * r . F. Vm/ (RT) )
  // Ireverse = [Na]_o^3 * Ca_i * exp( -(n-2)* r * F. Vm/ (RT) )
  // INCX = k * (Iforward - Ireverse)   [pA/um^2]
  // ICa  = -zCa * (n-2) * INCX              [pA/um^2]
  // For 3Na:1Ca, i.e. n=3
  // ICa  = -2 * INCX              [pA/um^2]
  // 
  // In NTS:
  //   [Na]_i, [Na]_o [mM]
  //   [Ca]_o [uM]
  //   [Ca]_i [uM] 
  //   original paper
  //      NOTE using Cm=1.0 uF/cm^2
  //   k      [uA.mM^{-4}.cm^-2] or [uA.mM^{-4}.uF] 
  //       = k_NTS * 10^6 (pA) . (mM^-4) . (10^-8). um^-2
  //       = k_NTS/100 [pA.mM^-4.um^-2]
  //    so k_NTS = k*100
  // NOTE:
  //   k = 2.07e-5 uA.mM^-4.uF; r=0.38; ENaCa=-98mV (-100mV data)   for 0.1mM [Ca]_o
  //   k = 5.94(+/-2.74)e-5 uA.mM^-4.uF; ENaCa=-95.7mV   for 1.0mM [Ca]_o
  //   k = 6.28(+/-3.57)e-5 uA.mM^-4.uF; ENaCa=-100 +/- 8 mV (expected -119mV)  for 2.0mM [Ca]_o
  //
  //   r = 0.38 
#define eta_Na 3
#define eta_Ca 1
//#define NCX2Caconversion (zCa / (eta_Na * zNa - eta_Ca * zCa))
#define NCX2Caconversion 2
#define k_NCX 6.28e-3  // [pA/(mM^4.um^2)]
#define gamma_NCX 0.38
#define E_1 ((eta_Na-zCa*eta_Ca)*gamma_NCX * zF_RT)
#define E_2 (-(eta_Na-zCa*eta_Ca)*(1 - gamma_NCX) * zF_RT)

#elif EXCHANGER_NCX == NCX_Gabbiani_Midtgaard_Kopfel_1994
  // NOTE Gabbiani-Midtgaard-Knopfel (1994) Synaptic Integration in model of 
  //         cerebellar granule cell (PNAS)
  // Based on Kimura et al. (1987) model
#define NCX_FORMULA_BASED   NCX_KIMURA_BASED_FORMULA
  // NOTE: k is 10x higher than that used/estimated in Kimura et al. (1987) 
  //  k_NCX 4.677e-4  // [uA/(mM^4.cm^2)]
#define eta_Na 3
#define eta_Ca 1
//#define NCX2Caconversion (zCa / (eta_Na * zNa - eta_Ca * zCa))
#define NCX2Caconversion 2
// NOTE: mM unit is used for this model
#define k_NCX 4.677e-2  // [pA/(mM^4.um^2)]
#define gamma_NCX 0.38
//#define E_1 ((eta_Na-2)*gamma_NCX * zF_RT)
//#define E_2 (-(eta_Na-2)*(1 - gamma_NCX) * zF_RT)
//#define E_1 ((eta_Na-zCa*eta_Ca)*gamma_NCX * zF_RT)
//#define E_2 (-(eta_Na-zCa*eta_Ca)*(1 - gamma_NCX) * zF_RT)
#define E_1 (0.013)   # 1/mV
#define E_2 (-0.026)   # 1/mV

#elif EXCHANGER_NCX == NCX_Weber_Bers_2001
// NOTE: This model ensures a saturation in the current 
//     at high Vm and saturation at high [Ca]_i
#define NCX_FORMULA_BASED   NCX_WEBER_BASED_FORMULA
//
// INCX = (AlloSteric_Factor) * (ElectroChemical_Factor)
// AlloSteric_Factor  = 1.0 / (1 + (K_mCaact / [Ca]_i)^nHill) 
// ElectroChemical_Factor = V_max { Iforward - Ibackward} / { Factor_1 * Factor_2 } 
// Factor_1 = KmCao * Nai^3 + KmNao^3 * Cai + KmNai^3 * Cao * (1 + Cai / KmCai) + \
//            KmCai * Nao^3 (1 + Nai^3 / KmNai^3) + Nai^3 * Cao + Nao^3 * Cai 
// Factor_2 = 1 + ksat * exp(E_2 * V)
#define eta_Na 3
#define eta_Ca 1
// NOTE: a typically Ica current has 1Ca exchange generates zCa=2 unit current
// so JCa = Ica * A / (zCa * F * Vmyo)
//    1Ca exchange generates 1 unit current
// so JCa(NCX)= JCa * zCa
//#define NCX2Caconversion (zCa / (eta_Na * zNa - eta_Ca * zCa))
#define NCX2Caconversion 2
#define gamma_NCX 0.38  // energy partition factor
#define KNCX_Ca 1380    // [uM]
#define KNCX_Na 87500   // [uM]
#define ksat_NCX 0.1    // NCX saturation factor

#elif EXCHANGER_NCX == _COMPONENT_UNDEFINED
   // do nothing
#endif

void ExchangerNCX::initialize(RNG& rng)
{
//  pthread_once(&once_Exchanger_NCX, initialize_others);
#ifdef DEBUG_ASSERT
  assert(branchData);
#endif
  unsigned size = branchData->size;
#ifdef DEBUG_ASSERT
  assert(V);
#if EXCHANGER_NCX == NCX_Weber_Bers_2001
  assert(INCXbar.size() == size);
#endif
  assert(V->size() == size);
#endif
  // allocate
  if (I_NCX.size() != size) I_NCX.increaseSizeTo(size);
  if (I_Ca.size() != size) I_Ca.increaseSizeTo(size);
#ifdef CONSIDER_DI_DV
  if (conductance_didv.size() != size) conductance_didv.increaseSizeTo(size);
#endif
  // initialize
#if EXCHANGER_NCX == NCX_Weber_Bers_2001
  if (INCXbar.size() != size) INCXbar.increaseSizeTo(size);
  dyn_var_t INCXbar_default = INCXbar[0];
  if (INCXbar_dists.size() > 0 and INCXbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either INCXbar_dists or INCXbar_branchorders on "
                 "GHK-formula Ca2+ channel "
                 "Channels Param" << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
    if (INCXbar_dists.size() > 0)
    {
      unsigned int j;
      assert(INCXbar_values.size() == INCXbar_dists.size());
      for (j = 0; j < INCXbar_dists.size(); ++j)
      {
        if ((*dimensions)[i]->dist2soma < INCXbar_dists[j]) break;
      }
      if (j < INCXbar_values.size())
        INCXbar[i] = INCXbar_values[j];
      else
        INCXbar[i] = INCXbar_default;
    }
    /*else if (INCXbar_values.size() == 1) {
INCXbar[i] = INCXbar_values[0];
} */
    else if (INCXbar_branchorders.size() > 0)
    {
      int j;
      assert(INCXbar_values.size() == INCXbar_branchorders.size());
      SegmentDescriptor segmentDescriptor;
      // const special_value = -1;
      for (j = 0; j < INCXbar_branchorders.size(); ++j)
      {
        if (segmentDescriptor.getBranchOrder(branchData->key) ==
            INCXbar_branchorders[j])
          break;
      }
      // if (j == INCXbar_branchorders.size() and INCXbar_branchorders[j-1] ==
      // special_value)
      if (j == INCXbar_branchorders.size() and
          INCXbar_branchorders[j - 1] == GlobalNTS::anybranch_at_end)
      {
        INCXbar[i] = INCXbar_values[j - 1];
      }
      else if (j < INCXbar_values.size())
        INCXbar[i] = INCXbar_values[j];
      else
        INCXbar[i] = INCXbar_default;
    }
    else
    {
      INCXbar[i] = INCXbar_default;
    }
  }
#endif

  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];  // mV
#ifdef CONSIDER_DI_DV
    conductance_didv[i] = 0;
#endif
#if EXCHANGER_NCX == NCX_Gabbiani_Midtgaard_Kopfel_1994 || \
    EXCHANGER_NCX == NCX_Kimura_Miyamae_Noma_1987
    I_NCX[i] = update_current(v, i);
    // must be opposite sign
    I_Ca[i] = -NCX2Caconversion * I_NCX[i];  // [pA/um^2]

#elif EXCHANGER_NCX == NCX_Weber_Bers_2001
    dyn_var_t cai = (*Ca_IC)[i];  //[uM]
    // The saturation is incorporated in the denominator of the formula
    I_NCX[i] = update_current(v, i);

    // must be opposite sign
    I_Ca[i] = -NCX2Caconversion * I_NCX[i];  // [pA/um^2]

#elif EXCHANGER_NCX == _COMPONENT_UNDEFINED
		// do nothing
#else
    NOT IMPLEMENTED YET;
#endif
  }
}

void ExchangerNCX::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if EXCHANGER_NCX == NCX_Gabbiani_Midtgaard_Kopfel_1994 || \
    EXCHANGER_NCX == NCX_Kimura_Miyamae_Noma_1987
    {
      // NOTE: k_NCX [pA. mM^-4 . um^-2], Vm [mV], Cai/o [uM], F [C/mol] or
      // [mJ/(mV.mol)]
      //     R [mJ/(mol.K)]
      I_NCX[i] = update_current(v, i);
      // must be opposite sign
      I_Ca[i] = -NCX2Caconversion * I_NCX[i];  // [pA/um^2]

#ifdef CONSIDER_DI_DV
      dyn_var_t I_NCX_dv = update_current(v+0.001, i);
      conductance_didv[i] = (I_NCX_dv-I_NCX[i])/(0.001);
#endif

    }
#elif EXCHANGER_NCX == NCX_Weber_Bers_2001
    {
      // The saturation is incorporated in the denominator of the formula
      I_NCX[i] = update_current(v, i);
      // must be opposite sign
      I_Ca[i] = -NCX2Caconversion * I_NCX[i];  // [pA/um^2]

#ifdef CONSIDER_DI_DV
      dyn_var_t I_NCX_didv = update_current(v+0.001, i);
      conductance_didv[i] = (I_NCX_didv-I_NCX[i])/(0.001);
#endif
    }
#elif EXCHANGER_NCX == _COMPONENT_UNDEFINED
    // do nothing
#endif

    /*
     * TUAN TODO: think about stochastic modelling
     * Incx[i] = Nncx * pumping_rate_per_molecule * ...
    */
  }
}

// v = voltage (mV)
// i = compartment index
dyn_var_t ExchangerNCX::update_current(dyn_var_t v, int i)
{// voltage v (mV) and return current density I_NCX(pA/um^2)
    // NOTE: k_NCX [pA. mM^-4 . um^-2], Vm [mV], Cai/o [uM], 
    //       F [C/mol] or [mJ/(mV.mol)]
    //       R [mJ/(mol.K)]
  dyn_var_t result = 0.0;
#if NCX_FORMULA_BASED == NCX_KIMURA_BASED_FORMULA
  dyn_var_t cai = (*Ca_IC)[i] * uM2mM;  //[mM]
  result = k_NCX * (*(getSharedMembers().Ca_EC) * uM2mM *
      pow(*(getSharedMembers().Na_IC), eta_Na) *
      exp(E_1 * v) -
      cai * pow(*(getSharedMembers().Na_EC), eta_Na) *
      exp(E_2 * v));
#elif NCX_FORMULA_BASED == NCX_WEBER_BASED_FORMULA
  dyn_var_t cai = (*Ca_IC)[i];  //[uM]
  result = INCXbar[i] *
    (*(getSharedMembers().Ca_EC) *
     pow(*(getSharedMembers().Na_IC) * mM2uM, eta_Na) *
     exp(gamma_NCX * zF_RT * v) -
     cai * pow(*(getSharedMembers().Na_EC) * mM2uM, eta_Na) *
     exp(-(1 - gamma_NCX) * zF_RT * v)) /
    ((pow(KNCX_Na, eta_Na) +
      pow(*(getSharedMembers().Na_EC) * mM2uM, eta_Na)) *
     (KNCX_Ca + *(getSharedMembers().Ca_EC)) *
     (1 + ksat_NCX * exp((gamma_NCX - 1) * zF_RT * v)));

#else
  assert(0);
#endif
  return result;
} 

ExchangerNCX::~ExchangerNCX() {}
