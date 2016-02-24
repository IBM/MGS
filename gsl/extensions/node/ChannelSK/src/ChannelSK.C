#include "Lens.h"
#include "ChannelSK.h"
#include "CG_ChannelSK.h"
#include "rndm.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

//
// This is an implementation of the "SK(ca) potassium current
//
#if CHANNEL_SK == SK_WOLF_2005
// The implementation is indeed from Moczydlowski - Latorre 1993
//    for Ca-dependent only K+ channel from rat skeletal muscle
//    (single channel current)
//
//  Use Model 3 from the paper: C <->C.Ca <=> O.Ca <-> O.Ca2
//   C = C, C.Ca
//   O = O.Ca, O.Ca2
//  which is mapped to Hodgkin-Huxley-type formula using
//  steady-state modeling:
//     (C,C.Ca) <=>[fwrate][bwrate] (O.Ca, O.Ca2)
//     only look at Po at equilibrium
//     Po = fwrate / (fwrate + bwrate)
#define alpha 0.28  // [1/ms]
#define beta 0.480  // [1/ms]
#else
NOT IMPLEMENTED YET
#endif

// GOAL: find K(V) = K(0) * exp(-zCa * delta * FV / (RT))
//   - dissociation constant as a function of voltage
// Unit on return: mM
// k(mM), d=delta, v(mV)
// F=Faraday
// R=universial constant
// T=temperature
dyn_var_t ChannelSK::KV(dyn_var_t k, dyn_var_t d, dyn_var_t v)
{
  //const dyn_var_t zCa = 2;
  dyn_var_t exp1 = k * exp(-zCa * d * zF * v / zR / (*(getSharedMembers().T)));
  return exp1;
}

dyn_var_t  ChannelSK::fwrate(dyn_var_t v, dyn_var_t cai)
{
	dyn_var_t fwrate;
	const dyn_var_t d1 = 0.84;
	const dyn_var_t k1 = 0.18; //[mM] - K(0)
	fwrate = beta / (1 + KV(k1,d1,v)/cai);
	return fwrate;
}
dyn_var_t ChannelSK::bwrate(dyn_var_t v, dyn_var_t cai)
{
	dyn_var_t bwrate;
	const dyn_var_t d2 = 1.0;
	const dyn_var_t k2 = 0.011; //[mM] - K(0)
	bwrate = alpha / (1 + cai/KV(k2,d2,v));
	return bwrate;
}

void ChannelSK::update(RNG& rng)
{
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

#if CHANNEL_SK == SK_WOLF_2005
    // Rate k1-k2: unit 1/(ms)

    dyn_var_t Oval = fO[i];  // temporary
                             // Rempe-Chopp 2006
		dyn_var_t a = fwrate(v, cai);
		dyn_var_t sum = a+bwrate(v,cai);
		dyn_var_t tau = 1/(sum);
		dyn_var_t Oinf = a/(sum); 
		dyn_var_t Tscale_tau = dt  * getSharedMembers().Tadj /tau;
		fO[i] = (Oinf * Tscale_tau + Oval * (1-Tscale_tau))/(1+Tscale_tau);
#ifdef DEBUG_ASSERT
    //assert(fabs(fO[0] + fC[0] - 1.0) < SMALL);  // conservation
#endif
#else
    NOT IMPLEMENTED YET
#endif
    // trick to keep fO in [0, 1]
    if (fO[i] < 0.0) { fO[i] = 0.0; }
    else if (fO[i] > 1.0) { fO[i] = 1.0; }
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
    // fC[i] = 1.0 - (fO[i]);  //no need
    g[i] = gbar[i] * fO[i] ;
  }
}

void ChannelSK::initialize(RNG& rng)
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
  //if (fC.size() != size) fC.increaseSizeTo(size);
  // initialize
  //assert(fabs(fO[0] + fC[0] - 1.0) < SMALL);  // conservation
  for (unsigned i = 0; i < size; ++i)
  {
    gbar[i] = gbar[0];
    //fO[i] = fO[0];
    dyn_var_t v = (*V)[i];
    //g[i] = gbar[i] * fO[i];
#if SIMULATION_INVOLVE  == VMONLY
		dyn_var_t Cai_base = 0.1e-3; // [mM]
    dyn_var_t cai = Cai_base;
#else
		dyn_var_t cai = (*Cai)[i];
#endif
		dyn_var_t a = fwrate(v, cai);
		dyn_var_t sum = a+bwrate(v,cai);
		dyn_var_t tau = 1/(sum);
		dyn_var_t Oinf = a/(sum); 
		fO[i] = Oinf;
  }
}

ChannelSK::~ChannelSK() {}
