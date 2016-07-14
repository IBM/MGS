#include "Lens.h"
#include "ChannelSK.h"
#include "CG_ChannelSK.h"
#include "rndm.h"

#define SMALL 1.0E-6
// unit conversion 
#define uM2mM  1e-3 // from [uM] to [mM] concentration

#include <math.h>
#include <pthread.h>
#include <algorithm>

#define Cai_base 0.1e-6 // [uM]
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

#elif CHANNEL_SK == SK2_KOHLER_ADELMAN_1996_RAT
// Kohler -...-Adelman (1996) Science 
//   Small-conductance, Ca2+ activated potassium channels from mammalian brain
//   1. SK permeate to K+ rather than Na+
//   2. record data for hSK1 and rSK2
//   3. current activate instantly and show no inactivation (during 500ms test pulse)
//   SK2 and SK3 is apamin-sensitive
//   SK1 is apamin-insensitive
//   rSK2: gmax = 3.6 [nS]
//   hSK1: gmax = 13.5 [nS]
//  NOTE: Hill_coeff == steepness
#define KCa_half  0.43  // [uM]
#define Hill_coef  4.8 // 4.8+/-1.46  
#define TAU 1.0

#elif CHANNEL_SK == SK1_KOHLER_ADELMAN_1996_HUMAN
#define KCa_half  0.71  // [uM]
#define Hill_coef  3.9  // 3.9+/-0.45 suggest 4 Ca2+ binding sites involved
#define TAU 1.0

#elif CHANNEL_SK == SK_TRAUB_1994
// SK equation used in many Traub papers:
//   alpha = min(0.2x10^-4 x [Ca]i, 0.01)
//   beta = 0.001
//   chi means [Ca]i
#define alphamax 0.01
#define chiscaling 0.00002
#define beta 0.001
#else
NOT IMPLEMENTED YET
#endif

#ifndef  alpha
#define alpha 0.00001  // [1/ms]
#endif
#ifndef beta
#define beta 0.000010  // [1/ms]
#endif
// GOAL: find K(V) = K(0) * exp(-zCa * delta * FV / (RT))
//   - dissociation constant as a function of voltage
// Unit on return: mM
// k(mM), d=delta, v(mV)
// F=Faraday
// R=universial constant
// T=temperature
// d=delta=fractional distance of the electric field that is felt by the ions
dyn_var_t ChannelSK::KV(dyn_var_t k, dyn_var_t d, dyn_var_t v)
{
  //const dyn_var_t zCa = 2;
  dyn_var_t exp1 = k * exp(-zCa * d * zF * v / zR / (*(getSharedMembers().T)));
  return exp1;
}

// NOTE: 
//    v = voltage [mV]
//    cai = [Ca]cyto [mM]
dyn_var_t  ChannelSK::fwrate(dyn_var_t v, dyn_var_t cai)
{
	dyn_var_t fwrate;
	const dyn_var_t d1 = 0.84; // [0..1] unitless
	const dyn_var_t k1 = 0.18; //[mM] - K(0)
	fwrate = beta / (1 + KV(k1,d1,v)/cai);
	return fwrate;
}
dyn_var_t ChannelSK::bwrate(dyn_var_t v, dyn_var_t cai)
{
	dyn_var_t bwrate;
	const dyn_var_t d2 = 1.0;// [0..1] unitless
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
    dyn_var_t cai = Cai_base;
#else
    dyn_var_t cai = (*Cai)[i] ;  //[uM]
#endif

#if CHANNEL_SK == SK_WOLF_2005
	{//to group code
    // Rate k1-k2: unit 1/(ms)
	dyn_var_t cai_mM = cai * uM2mM; //[mM]

    dyn_var_t Oval = fO[i];  // temporary
	// Rempe-Chopp 2006
	dyn_var_t a = fwrate(v, cai_mM);
	dyn_var_t sum = a+bwrate(v,cai_mM);
	dyn_var_t tau = 1/(sum);
	dyn_var_t Oinf = a/(sum); 
	dyn_var_t Tscale_tau = dt  * getSharedMembers().Tadj /tau;
	fO[i] = (Oinf * Tscale_tau + Oval * (1-Tscale_tau))/(1+Tscale_tau);
#ifdef DEBUG_ASSERT
    //assert(fabs(fO[0] + fC[0] - 1.0) < SMALL);  // conservation
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

	}
    g[i] = gbar[i] * fO[i] ;
#elif CHANNEL_SK == SK2_KOHLER_ADELMAN_1996_RAT || \
	  CHANNEL_SK == SK1_KOHLER_ADELMAN_1996_HUMAN
	// Rempe-Chopp 2006
	dyn_var_t minf = (1.0)/ (1.0 + pow(KCa_half/cai, Hill_coef));
	dyn_var_t qm = 0.5 * dt * getSharedMembers().Tadj/ TAU ;
	//fO means 'm' (activation gate)
    //m[i] = (2 * minf * qm - m[i] * (qm-1)) / (qm + 1);
    //g[i] = gbar[i]*m[i];
    fO[i] = (2 * minf * qm - fO[i] * (qm-1)) / (qm + 1);
    g[i] = gbar[i]*fO[i];

#elif CHANNEL_SK == SK_TRAUB_1994
    // a is alpha
    dyn_var_t aq = ((chiscaling*cai)>alphamax)?alphamax:(chiscaling*cai);
    dyn_var_t bq = beta;
    // Rempe & Chopp (2006)
    dyn_var_t pq = 0.5*dt*(aq+bq);
    fO[i] = (dt*aq + fO[i]*(1.0-pq))/(1.0+pq);
    g[i] = gbar[i]*fO[i];
#else
    NOT IMPLEMENTED YET
#endif
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
	dyn_var_t Cai_base = 0.1e-6; // [uM]
    dyn_var_t cai = Cai_base;
#else
	dyn_var_t cai = (*Cai)[i]; // [uM]
#endif

#if CHANNEL_SK == SK_WOLF_2005
	dyn_var_t cai_mM = cai  * uM2mM; // [mM]
	dyn_var_t a = fwrate(v, cai_mM);
	dyn_var_t sum = a+bwrate(v,cai_mM);
	dyn_var_t tau = 1/(sum);
	dyn_var_t Oinf = a/(sum); 
	fO[i] = Oinf;
    g[i] = gbar[i]*fO[i];
#elif CHANNEL_SK == SK2_KOHLER_ADELMAN_1996_RAT || \
	  CHANNEL_SK == SK1_KOHLER_ADELMAN_1996_HUMAN
    //m[i] = 1.0/(1 + pow(KCa_half/cai,Hill_coef));
    //g[i] = gbar[i]*m[i];
    fO[i] = 1.0/(1 + pow(KCa_half/cai,Hill_coef));
    g[i] = gbar[i]*fO[i];
#elif CHANNEL_SK == SK_TRAUB_1994
    // a is alpha
    dyn_var_t aq = ((chiscaling*cai)>alphamax)?alphamax:(chiscaling*cai);
    dyn_var_t bq = beta;
    fO[i] = aq/(aq+bq);
    g[i] = gbar[i]*fO[i];
#endif
  }
}

ChannelSK::~ChannelSK() {}
