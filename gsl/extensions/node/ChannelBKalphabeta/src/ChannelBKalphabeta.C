// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "ChannelBKalphabeta.h"
#include "CG_ChannelBKalphabeta.h"
#include "rndm.h"

#define SMALL 1.0E-6
// unit conversion 
#ifndef uM2mM
#define uM2mM  1e-3
#endif

#include <math.h>
#include <pthread.h>
#include <algorithm>

//
// This is an implementation of the "BK(ca) potassium current
//      with alpha+beta subunit (fast-activating, and inactivating K+ current)
//
#if CHANNEL_BKalphabeta == BKalphabeta_SHAO_1999 || \
	CHANNEL_BKalphabeta == BKalphabeta_WOLF_2005
// The implementation is indeed from Shao et al. 1999
//    for CA1 pyramidal neuron
//    alpha+beta-generic
//    Original Borg-Graham (1998): Vm-dependent inactivation speed up with
//    hyperpolarization
//       -> inactivate during interspike, but not during sustained depolarizing
//       step
//    New: also inactivate during sustained depolarizing step
//    NOTE: Both lack a 'shoulder' suggest it miss a kinetic component
//       making the frequency-dependent broadening of AP is less thann observed
//
//  Markov-based model with 3-state (C,O,I)
// C <-> O   [k3, k4]
// O <-> I   [k1, 0.0]
// I <-> C   [k2, 0.0]
#else
NOT IMPLEMENTED YET
#endif
// GOAL: return the rate  (1/ms)
//( tmin(ms), tmax(ms), v(mV), vhalf(mV), k(mV)  )(/ms){
// 1/(tmin + 1/(1/(tmax-tmin) + exp((V-Vhalf)/k)))
dyn_var_t ChannelBKalphabeta::alpha(dyn_var_t tmin, dyn_var_t tmax, dyn_var_t v,
                                    dyn_var_t vhalf, dyn_var_t k)
{
  dyn_var_t alpha;
  const dyn_var_t q = 1.0;     // unitless
  const dyn_var_t koff = 1.0;  // (1/ms)
  alpha =
      q / (tmin + 1.0 / (1.0 / (tmax - tmin) + exp((v - vhalf) / k) * koff));
  return alpha;
}

// GOAL: return the rate (1/ms)
//  when 1/tmax = 0
//(tmin(ms), v(mV), vhalf(mV), k(mV))(/ ms)
dyn_var_t ChannelBKalphabeta::alp(dyn_var_t tmin, dyn_var_t v, dyn_var_t vhalf,
                                  dyn_var_t k)
{
  dyn_var_t alp;
  const dyn_var_t q = 1.0;     // unitless
  const dyn_var_t koff = 1.0;  // (1/ms)
  alp = q / (tmin + 1.0 / (exp((v - vhalf) / k) * koff));
  return alp;
}

void ChannelBKalphabeta::initialize(RNG& rng)
{
  assert(branchData);
  unsigned int size = branchData->size;
  if (not V)
  {
    std::cerr << typeid(*this).name() << " needs Voltage as input in ChanParam\n";
    assert(V);
  }
#if defined(SIMULATE_CACYTO)
  if (not Cai)
  {
    std::cerr << typeid(*this).name() << " needs Calcium as input in ChanParam\n";
    assert(Cai);
  }
#endif
  assert(gbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (fO.size() != size) fO.increaseSizeTo(size);
  if (fI.size() != size) fI.increaseSizeTo(size);
  if (fC.size() != size) fC.increaseSizeTo(size);
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
  // initialize
  fI[0] = fO[0]= 0.0;
  assert(fabs(fI[0] + fO[0] + fC[0] - 1.0) < SMALL);  // conservation
  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];      //[mV]
    gbar[i] = gbar[0];
    fI[i] = fI[0];
    fO[i] = fO[0];
    fC[i] = fI[0];
    g[i] = gbar[i] * fO[i];
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
  }
}

void ChannelBKalphabeta::update(RNG& rng)
{
#if CHANNEL_BKalphabeta == BKalphabeta_SHAO_1999 || \
	CHANNEL_BKalphabeta == BKalphabeta_WOLF_2005
  const dyn_var_t tminOI = 0.1; //msec
  const dyn_var_t tminIC = 0.1; //msec
  const dyn_var_t tminCO = 0.001; //msec
  const dyn_var_t tminOC = 0.01; //msec
  const dyn_var_t tmaxCO = 1.0; //msec
  const dyn_var_t kV_OI = 1.0; //mV   = steepness of activation
  const dyn_var_t kV_IC = -10.0; //mV
  const dyn_var_t kV_CO = 7.0; //mV
  const dyn_var_t kV_OC = -5.0; //mV
  const dyn_var_t Vhalf_OI = -10.0; //mV
  const dyn_var_t Vhalf_IC = -120.0; //mV
  const dyn_var_t Vhalf_CO = -20.0; //mV
  const dyn_var_t Vhalf_OC = -44.0; //mV
#endif

  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];      //[mV]
		// IMPORTANT: Make sure to convert [Ca]cyto from [uM] to [mM]
#if ! defined(SIMULATE_CACYTO)
    dyn_var_t Cai_base = 0.1; // [uM]
    dyn_var_t cai = Cai_base * uM2mM;
#else
    dyn_var_t cai = (*Cai)[i] * uM2mM;  //[mM]
#endif

#if CHANNEL_BKalphabeta == BKalphabeta_SHAO_1999 || \
	CHANNEL_BKalphabeta == BKalphabeta_WOLF_2005
    // NOTE: Some models use m_inf and tau_m to estimate m
    // Rate kij from state i to j: unit 1/(ms)
    dyn_var_t kOI = alp(tminOI, v, Vhalf_OI, kV_OI);
    dyn_var_t kIC = alp(tminIC, v, Vhalf_IC, kV_IC);
    const dyn_var_t Ca_factor = 1.0e8;  // [1/(mM^n)] - proportional to unit
    const dyn_var_t n = 3; // the number of bound Ca2+ ions
    dyn_var_t Ca_depdency = Ca_factor * pow(cai, n);
    dyn_var_t kCO = alpha(tminCO, tmaxCO, v, Vhalf_CO, kV_CO) * Ca_depdency;
    dyn_var_t kOC = alp(tminOC, v, Vhalf_OC, kV_OC);

    dyn_var_t Oval = fO[i], Cval = fC[i], Ival = fI[i];  // temp
                                                         // Rempe-Chopp 2006
    // dfO[i] = Cval*kCO-Oval*(kOC+kOI);
    dyn_var_t Tscale = dt ;// * getSharedMembers().Tadj;
    fO[i] = ((Cval * kCO - Oval * (kOC + kOI) / 2) * Tscale + Oval) /
            (1 + (kOC + kOI) / 2 * Tscale);
    // dfI[i] = Oval*kOI-Ival*kIC;
    fI[i] =
        ((Oval * kOI - Ival * kIC / 2) * Tscale + Ival) / (1 + kIC / 2 * Tscale);
#else
    NOT IMPLEMENTED YET
#endif
    // trick to keep fO in [0, 1]
    if (fO[i] < 0.0) { fO[i] = 0.0; }
    else if (fO[i] > 1.0) { fO[i] = 1.0; }
    // trick to keep fI in [0, 1]
    if (fI[i] < 0.0) { fI[i] = 0.0; }
    else if (fI[i] > 1.0) { fI[i] = 1.0; }
    // trick to keep fC in [0, 1]
    //if (fC[i] < 0.0) { fC[i] = 0.0; }
    //else if (fC[i] > 1.0) { fC[i] = 1.0; }
    fC[i] = 1.0 - (fO[i] + fI[i]);
#ifdef DEBUG_LOOPS
#ifdef DEBUG_HH 
		if (fabs(fI[i] + fO[i] + fC[i] - 1.0) < SMALL)
		{
			std::cerr << i << ": fI =" << fI[i] << ",fO = " << fO[i] << ", fC =" << fC[i] << std::endl;
		}
#endif
#endif
#ifdef DEBUG_ASSERT
    assert(fabs(fI[i] + fO[i] + fC[i] - 1.0) < SMALL);  // conservation
#endif
    g[i] = gbar[i] * fO[i] ;
		Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
  }
}

ChannelBKalphabeta::~ChannelBKalphabeta() {}
