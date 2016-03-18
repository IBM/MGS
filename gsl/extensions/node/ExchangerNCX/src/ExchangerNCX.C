#include "Lens.h"
#include "ExchangerNCX.h"
#include "CG_ExchangerNCX.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"
#include "NTSMacros.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

#if EXCHANGER_NCX == NCX_Gabbiani_Midtgaard_Kopfel_1994
  // NOTE Gabbiani-Midtgaard-Knopfel (1994) Synaptic Integration in model of 
	//         cerebellar granule cell (PNAS)
	//  
#define eta_Na 3
#define eta_Ca 1
//#define NCX2Caconversion (zCa / (eta_Na * zNa - eta_Ca * zCa))
#define NCX2Caconversion 2
// NOTE: mM unit is used for this model
#define k_NCX 4.677e-6  // [pA/(mM^4.um^2)]
#define gamma_NCX 0.38

#elif EXCHANGER_NCX == NCX_Weber_Bers_2001
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
#else

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
  assert(INCXbar.size() == size);
  assert(V->size() == size);
#endif
  // allocate
  if (I_NCX.size() != size) I_NCX.increaseSizeTo(size);
  if (I_Ca.size() != size) I_Ca.increaseSizeTo(size);
  // initialize
#if EXCHANGER_NCX == NCX_Weber_Bers_2001
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
#if EXCHANGER_NCX == NCX_Gabbiani_Midtgaard_Kopfel_1994
    // NOTE: k_NCX [pA. mM^-4 . um^-2], Vm [mV], Cai/o [uM], F [C/mol] or
    // [mJ/(mV.mol)]
    //     R [mJ/(mol.K)]
    dyn_var_t cai = (*Ca_IC)[i] * 1e-3;  //[mM]
    // There is no saturation in this model
    I_NCX[i] = k_NCX * (*(getSharedMembers().Ca_EC) * uM2mM *
                            pow(*(getSharedMembers().Na_IC), eta_Na) *
                            exp(gamma_NCX * zF_RT * v) -
                        cai * pow(*(getSharedMembers().Na_EC), eta_Na) *
                            exp(-(1 - gamma_NCX) * zF_RT * v));
    // must be opposite sign
    I_Ca[i] = -NCX2Caconversion * I_NCX[i];  // [pA/um^2]

#elif EXCHANGER_NCX == NCX_Weber_Bers_2001
    dyn_var_t cai = (*Ca_IC)[i];  //[uM]
    // The saturation is incorporated in the denominator of the formula
    I_NCX[i] = INCXbar[i] *
               (*(getSharedMembers().Ca_EC) *
                    pow(*(getSharedMembers().Na_IC) * mM2uM, eta_Na) *
                    exp(gamma_NCX * zF_RT * v) -
                cai * pow(*(getSharedMembers().Na_EC) * mM2uM, eta_Na) *
                    exp(-(1 - gamma_NCX) * zF_RT * v)) /
               ((pow(KNCX_Na, eta_Na) +
                 pow(*(getSharedMembers().Na_EC) * mM2uM, eta_Na)) *
                (KNCX_Ca + *(getSharedMembers().Ca_EC)) *
                (1 + ksat_NCX * exp((gamma_NCX - 1) * zF_RT * v)));

    // must be opposite sign
    I_Ca[i] = -NCX2Caconversion * I_NCX[i];  // [pA/um^2]

#elif EXCHANGER_NCX == _COMPONENT_UNDEFINED
		// do nothing
#else
    NOT IMPLEMENTED YET
#endif
  }
}

void ExchangerNCX::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if EXCHANGER_NCX == NCX_Gabbiani_Midtgaard_Kopfel_1994
    // NOTE: k_NCX [pA. mM^-4 . um^-2], Vm [mV], Cai/o [uM], F [C/mol] or
    // [mJ/(mV.mol)]
    //     R [mJ/(mol.K)]
    dyn_var_t cai = (*Ca_IC)[i] * 1e-3;  //[mM]
    // There is no saturation in this model
    I_NCX[i] = k_NCX * (*(getSharedMembers().Ca_EC) * uM2mM *
                            pow(*(getSharedMembers().Na_IC), eta_Na) *
                            exp(gamma_NCX * zF_RT * v) -
                        cai * pow(*(getSharedMembers().Na_EC), eta_Na) *
                            exp(-(1 - gamma_NCX) * zF_RT * v));
    // must be opposite sign
    I_Ca[i] = -NCX2Caconversion * I_NCX[i];  // [pA/um^2]

#elif EXCHANGER_NCX == NCX_Weber_Bers_2001
    dyn_var_t cai = (*Ca_IC)[i];  //[uM]
    // The saturation is incorporated in the denominator of the formula
    I_NCX[i] = INCXbar[i] *
               (*(getSharedMembers().Ca_EC) *
                    pow(*(getSharedMembers().Na_IC) * mM2uM, eta_Na) *
                    exp(gamma_NCX * zF_RT * v) -
                cai * pow(*(getSharedMembers().Na_EC) * mM2uM, eta_Na) *
                    exp(-(1 - gamma_NCX) * zF_RT * v)) /
               ((pow(KNCX_Na, eta_Na) +
                 pow(*(getSharedMembers().Na_EC) * mM2uM, eta_Na)) *
                (KNCX_Ca + *(getSharedMembers().Ca_EC)) *
                (1 + ksat_NCX * exp((gamma_NCX - 1) * zF_RT * v)));

    // must be opposite sign
    I_Ca[i] = -NCX2Caconversion * I_NCX[i];  // [pA/um^2]
#elif EXCHANGER_NCX == _COMPONENT_UNDEFINED
		// do nothing
#else
		NOT IMPLEMENTED YET
#endif

    /*
     * TUAN TODO: think about stochastic modelling
     * Incx[i] = Nncx * pumping_rate_per_molecule * ...
    */
  }
}

ExchangerNCX::~ExchangerNCX() {}
