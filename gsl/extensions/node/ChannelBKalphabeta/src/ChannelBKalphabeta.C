#include "Lens.h"
#include "ChannelBKalphabeta.h"
#include "CG_ChannelBKalphabeta.h"
#include "rndm.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

//
// This is an implementation of the "BK(ca) potassium current
//      with alpha+beta subunit (fast-activating, and inactivating K+ current)
//
#if CHANNEL_BKalphabeta == BKalphabeta_SHAO_1999 || CHANNEL_BKalphabeta == BKalphabeta_WOLF_2005
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
#ifdef DEBUG_ASSERT
  assert(branchData);
#endif
  unsigned size = branchData->size;
#ifdef DEBUG_ASSERT
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
#endif
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (fO.size() != size) fO.increaseSizeTo(size);
  if (fI.size() != size) fI.increaseSizeTo(size);
  if (fC.size() != size) fC.increaseSizeTo(size);
  // initialize
  assert(fabs(fI[0] + fO[0] + fC[0] - 1.0) < SMALL);  // conservation
  for (unsigned i = 0; i < size; ++i)
  {
    gbar[i] = gbar[0];
    fI[i] = fI[0];
    fO[i] = fO[0];
    fC[i] = fI[0];
    dyn_var_t v = (*V)[i];
    g[i] = gbar[i] * fO[i];
  }
}

void ChannelBKalphabeta::update(RNG& rng)
{
	const dyn_var_t tminOI = 0.1; //msec
	const dyn_var_t tminIC = 0.1; //msec
	const dyn_var_t tminCO = 0.001; //msec
	const dyn_var_t tminOC = 0.01; //msec
	const dyn_var_t tmaxCO = 1; //msec
	const dyn_var_t kOI = 1; //mV   = steepness of activation
	const dyn_var_t kIC = -10; //mV
	const dyn_var_t kCO = 7; //mV
	const dyn_var_t kOC = -5; //mV
	const dyn_var_t VhOI = -10.0; //mV
	const dyn_var_t VhIC = -120.0; //mV
	const dyn_var_t VhCO = -20.0; //mV
	const dyn_var_t VhOC = -44.0; //mV
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];      //[mV]
#if SIMULATION_INVOLVE  == VMONLY
		dyn_var_t Cai_base = 0.1e-3; // [mM]
    dyn_var_t cai = Cai_base;
#else
    dyn_var_t cai = (*Cai)[i];  //[mM]
#endif
#if CHANNEL_BKalphabeta == BKalphabeta_WOLF_2005
    // NOTE: Some models use m_inf and tau_m to estimate m
    // Rate kij from state i to j: unit 1/(ms)
    dyn_var_t kOI = alp(tminOI, v, VhOI, kOI);
    dyn_var_t kIC = alp(tminIC, v, VhIC, kIC);
    const dyn_var_t ca_factor = 1.0e8;  // [1/(mM^n)] - proportional to unit
    const dyn_var_t n = 3;
    dyn_var_t Ca_depdency = ca_factor * pow(cai, n);
    dyn_var_t kCO = alpha(tminCO, 1.0, v, VhCO, kCO) * Ca_depdency;
    dyn_var_t kOC = alp(tminOC, v, VhOC, kOC);

    dyn_var_t Oval = fO[i], Cval = fC[i], Ival = fI[i];  // temp
                                                         // Rempe-Chopp 2006
    // dfO[i] = Cval*kCO-Oval*(kOC+kOI);
    dyn_var_t Tscale = dt * getSharedMembers().Tadj;
    fO[i] = ((Cval * kCO - Oval * (kOC + kOI) / 2) * Tscale + Oval) /
            (1 + (kOC + kOI) / 2 * Tscale);
    // dfI[i] = Oval*kOI-Ival*kIC;
    fI[i] =
        ((Oval * kOI - Ival * kIC / 2) * Tscale + Ival) / (1 + kIC / 2 * Tscale);
#ifdef DEBUG_ASSERT
    assert(fabs(fI[0] + fO[0] + fC[0] - 1.0) < SMALL);  // conservation
#endif
#else
    NOT IMPLEMENTED YET
#endif
    // trick to keep fO in [0, 1]
    if (fO[i] < 0.0) { fO[i] = 0.0; }
    else if (fO[i] > 1.0) { fO[i] = 1.0; }
    // trick to keep fI in [0, 1]
    if (fI[i] < 0.0)
    {
      fI[i] = 0.0;
    }
    else if (fI[i] > 1.0)
    {
      fI[i] = 1.0;
    }
    // trick to keep fC in [0, 1]
    // if (fC[i] < 0.0)
    //{
    //	fC[i] = 0.0;
    //}
    // else if (fC[i] > 1.0)
    //{
    //	fC[i] = 1.0;
    //}
    // fC[i] = ...
    fC[i] = 1.0 - (fO[i] + fI[i]);
    g[i] = gbar[i] * fO[i] ;
  }
}

ChannelBKalphabeta::~ChannelBKalphabeta() {}
