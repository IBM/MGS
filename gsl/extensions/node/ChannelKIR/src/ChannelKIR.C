// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "ChannelKIR.h"
#include "CG_ChannelKIR.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"
#include "NumberUtils.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>
static pthread_once_t once_KIR = PTHREAD_ONCE_INIT;

//
// This is an implementation of the "KIR potassium current
//
#if CHANNEL_KIR == KIR_HAYASHI_FISHMAN_1988
// Article:
// Hayashi H, Fishman HM (1988). Inward rectifier K+ channel kinetics from
// analysis of the complex conductance of aplysia neuronal membrane.
// Biophys J 53, 747-757.
//
// NEURON : Aplysia neuron (non-amammalian)
// DATA: Vclamp in range -90 mV to -40 mV
//    1. Fig.8(B) - tau_m
//        IN Aplisia, tau_m is known to increase with increasing [K+]_o
//    2. Fig.8(A) - Vhalf and slope
//    3. Fig.2    - E_K in experiment
// CONDITION:
//    1. Ba2+ blockade (5mM) can block KIR
//    2. [K+]_o = 40 mM
//    3. Temperature = 22-24 celcius
//
//
// MODEL: 3-state Markov model  if [Ba2+] is considered (see ChannelKIR_Markov)
// MODEL: 2-state when assuming [Ba2+]_o is 0
#define Vmshift 0 // [mV] - to shift Vhalf if E_K from -45 in experiment is adjusted to E_K in the in vitro e.g. -90mV
#define VHALF_M (-52 + Vmshift)
#define k_M 13
#define LOOKUP_TAUM_LENGTH 6
const dyn_var_t ChannelKIR::_Vmrange_taum[] = {
    -90, -80, -70, -60, -50, -40};
dyn_var_t ChannelKIR::taumKIR[] = {
    4.0000, 4.7170, 5.3763, 6.0606, 6.8966, 7.6923};
std::vector<dyn_var_t> ChannelKIR::Vmrange_taum;

#elif CHANNEL_KIR == KIR_MAHON_2000
//assume activation is instantaneous
#define VHALF_M -100
#define k_M 10
//#define TAUM 1 //ms
#elif CHANNEL_KIR == KIR_WOLF_2005
// Model non-inactivating KIR (presumbly associated with SP(+) SPN in NAc) ~ Kir2.2???
// DATA:
//   1. steady-state activation curve:
//   2. time-constant
// DATA-1:
//  temperature 22-23 celcius
//  steady-state activation curve: use KIR channel (Kir2.1) data from
//     human embryonic kidney cells [Kubo, Murata, 2001] expressed in oocytes
//        --> this data use: [K]_i ~ 80 mMi and [K]_o = 10mM --> E_K ~ -52 mV
//   Fig.2(B) left
//
// DATA-2:
//  use DATA from non-inactivating KIR current ~ KIR2 family (here is Kir2.3)
//  YET it is adjusted as well - according to the value of [K+]_o in the model
//        IN Aplisia, tau_m is known to increase with increasing [K+]_o

/*
Uchimura N, Cherubini E, North RA (1989).  Inward rectification
in rat nucleus accumbens neurons. J Neurophysiol 62, 1280-1286.

Kubo Y, Murata Y (2001).  Control of rectification and permeation by two
distinct sites after the second transmembrane region in Kir2.1 K+
channel. J Physiol 531, 645-660.
*/

#define Vmshift -30 // [mV] - to shift Vhalf if E_K from -52 in experiment is adjusted to E_K in the in vitro e.g. -82mV
#define VHALF_M (-52 + Vmshift)
#define k_M 13
#define LOOKUP_TAUM_LENGTH 16
const dyn_var_t ChannelKIR::_Vmrange_taum[] = {
    -100, -90, -80, -70, -60, -50, -40, -30,
    -20, -10, 0, 10, 20, 30, 40, 50};
#if 0 // USE_NEURON_CODE == 1
//NOTE: The values here are used in NEURON-implementation which require Phi=0.5 (or Temperature = 43.320)
//dyn_var_t ChannelKIR::taumKIR[] = {
//    3.7313, 4.0000, 4.7170, 5.3763, 6.0606, 6.8966, 7.6923, 7.1429,
//    5.8824, 4.4444, 4.0000, 4.0000, 4.0000, 4.0000, 4.0000, 4.0000};

#else
//NOTE: The values here are exact values mapping to the 35 celcius
dyn_var_t ChannelKIR::taumKIR[] = {
    7.465,  8,      9.435 ,10.755, 12.12  ,13.795, 15.385, 14.285,
   11.765, 8.89,    8,      8,      8,      8,      8,      8};
#endif
std::vector<dyn_var_t> ChannelKIR::Vmrange_taum;

#elif CHANNEL_KIR == KIR_STEEPHEN_MANCHANDA_2009
// model non-inactivating Kir2.2 ???
// IMPORTANT: It fix Wolf-2005
//    1. Vmshift is applied to VHALF_M and tau_m in parallel. This fix the problem in Wolf-2005
//    2. Fix tau_m in the range [-120, -80] mV using data from Mermelstein, Surmeier (1998) Fig.1(C)
/*
Mermelstein PG, Song WJ, Tkatch T, Yan Z, Surmeier DJ (1998) Inwardly
rectifying potassium (IRK) currents are correlated with IRK subunit
expression in rat nucleus accumbens medium spiny neurons. J Neurosci
18:6650-6661.

 */
#define Vmshift -30 // [mV] - to adjust E_K from -52 in experiment to E_K in the in vitro
#define VHALF_M (-52 + Vmshift)
#define k_M 13
#define LOOKUP_TAUM_LENGTH 18
const dyn_var_t ChannelKIR::_Vmrange_taum[] = {-120, -110,
    -100, -90, -80, -70, -60, -50, -40, -30,
    -20, -10, 0, 10, 20, 30, 40, 50};
dyn_var_t ChannelKIR::taumKIR[] = {0.075, 0.219,
    0.691, 1.948 , 3.207, 10.755, 12.12  ,13.795, 15.385, 14.285,
   11.765, 8.89  ,    8,      8,      8,      8,      8,      8};
std::vector<dyn_var_t> ChannelKIR::Vmrange_taum;
#elif CHANNEL_KIR == KIR2_1_STEEPHEN_MANCHANDA_2009
// model inactivating Kir2.1
//  modified from those used in Wolf-2005
// IMPORTANT: It fix Wolf-2005
//    1. Vmshift is applied to VHALF_M and tau_m in parallel. This fix the problem in Wolf-2005
//    2. Fix tau_m in the range [-120, -80] mV using data from Mermelstein, Surmeier (1998) Fig.1(C)
//    3. Inactivation is incorporated Vm-dependent Fig.6 (C) Mermelstein, Surmeier (1998) for -90mV, -120mV
//         tau_h(-120) = 23.3ms; (-90mV) = 45ms; (-50mV) = 76ms
//        the time-constant was mapped to 22celcius; and need to convert to 35celcius
#define Vmshift -30 // [mV] - to adjust E_K from -52 in experiment to E_K in the in vitro
#define VHALF_M (-52 + Vmshift)
#define k_M 13
#define LOOKUP_TAUM_LENGTH 18
const dyn_var_t ChannelKIR::_Vmrange_taum[] = {-120, -110,
    -100, -90, -80, -70, -60, -50, -40, -30,
    -20, -10, 0, 10, 20, 30, 40, 50};
dyn_var_t ChannelKIR::taumKIR[] = {0.075, 0.219,
    0.691, 1.948 , 3.207, 10.755, 12.12  ,13.795, 15.385, 14.285,
   11.765, 8.89  ,    8,      8,      8,      8,      8,      8};
std::vector<dyn_var_t> ChannelKIR::Vmrange_taum;
#define LOOKUP_TAUH_LENGTH 3  // size of the below array
const dyn_var_t ChannelKIR::_Vmrange_tauh[] = {-120, -90, -50};
dyn_var_t ChannelKIR::tauhKIR[] = {7.767, 15, 25.33};
std::vector<dyn_var_t> ChannelKIR::Vmrange_tauh;
dyn_var_t ChannelKIR::h_inf_KIR[] = {0.0, 0.13, 1};

#elif CHANNEL_KIR == KIR2_1_TUAN_JAMES_2017
// model inactivating Kir2.1 (KIR1)
//     as found in substance ENK(+) and ENK/SP-expressing MSN in NAc [Mermelstein et al., 1998] - D1-MSN
// extended those in Steephen-2009
//  DATA:
//     Ariano, Levine (2004)
//        initial activation 	about -70 to -80 mV
//        Vhalf_activation -106 to -108 mV
//  time constant from Aplysia neuron (Kir2.3)
//
//  steady-state activation curve: use KIR channel (Kir2.1) data from
//    Ariano, Levine (2004) at -97 mV
/*

*/

#define Vmshift 7 // [mV] - to adjust E_K from -97 in experiment to E_K in the in vitro -90 mV
#define VHALF_M (-111 + Vmshift)
#define k_M 12.7
#define LOOKUP_TAUM_LENGTH 18
const dyn_var_t ChannelKIR::_Vmrange_taum[] = {-120, -110,
    -100, -90, -80, -70, -60, -50, -40, -30,
    -20, -10, 0, 10, 20, 30, 40, 50};
dyn_var_t ChannelKIR::taumKIR[] = {0.075, 0.219,
    0.691, 1.948 , 3.207, 10.755, 12.12  ,13.795, 15.385, 14.285,
   11.765, 8.89  ,    8,      8,      8,      8,      8,      8};
std::vector<dyn_var_t> ChannelKIR::Vmrange_taum;
#define LOOKUP_TAUH_LENGTH 3  // size of the below array
const dyn_var_t ChannelKIR::_Vmrange_tauh[] = {-120, -90, -50};
dyn_var_t ChannelKIR::tauhKIR[] = {7.767, 15, 25.33};
std::vector<dyn_var_t> ChannelKIR::Vmrange_tauh;

dyn_var_t ChannelKIR::h_inf_KIR[] = {0.0, 0.13, 1};

#else
NOT IMPLEMENTED YET
#endif


// GOAL: update gates using v(t+dt/2) and gate(t-dt/2)
//   --> output gate(t+dt/2)
//   of second-order accuracy at time (t+dt/2) using trapezoidal rule
void ChannelKIR::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_KIR == KIR_HAYASHI_FISHMAN_1988 || \
    CHANNEL_KIR == KIR_WOLF_2005 || \
    CHANNEL_KIR == KIR_STEEPHEN_MANCHANDA_2009
    // NOTE: Some models use m_inf and tau_m to estimate m
    std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
    int index = low - Vmrange_taum.begin();
    //dyn_var_t qm = dt * getSharedMembers().Tadj / (taumKIR[index] * 2);
    dyn_var_t taum;
    if (index == 0)
      taum = taumKIR[0];
    else
      taum = linear_interp(Vmrange_taum[index-1], taumKIR[index-1],
        Vmrange_taum[index], taumKIR[index], v);
    dyn_var_t qm = dt * getSharedMembers().Tadj / (taum * 2);

    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    g[i] = gbar[i] * m[i];
#elif CHANNEL_KIR == KIR2_1_STEEPHEN_MANCHANDA_2009 || \
      CHANNEL_KIR == KIR2_1_TUAN_JAMES_2017
    // NOTE: Some models use m_inf and tau_m to estimate m
    std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
    int index = low - Vmrange_taum.begin();
    //dyn_var_t qm = dt * getSharedMembers().Tadj / (taumKIR[index] * 2);
    dyn_var_t taum;
    if (index == 0)
      taum = taumKIR[0];
    else
      taum = linear_interp(Vmrange_taum[index-1], taumKIR[index-1],
        Vmrange_taum[index], taumKIR[index], v);
    dyn_var_t qm = dt * getSharedMembers().Tadj / (taum * 2);

    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);

    dyn_var_t tauh, h_inf;
    low = std::lower_bound(Vmrange_tauh.begin(), Vmrange_tauh.end(), v);
    index = low - Vmrange_tauh.begin();
    if (index == 0)
    {
      h_inf = h_inf_KIR[0];
      tauh = tauhKIR[0];
    }
    else
    {
      h_inf = linear_interp(Vmrange_tauh[index-1], h_inf_KIR[index-1],
        Vmrange_tauh[index], h_inf_KIR[index], v);
      tauh = linear_interp(Vmrange_tauh[index-1], tauhKIR[index-1],
        Vmrange_tauh[index], tauhKIR[index], v);
    }

    dyn_var_t qh = dt * getSharedMembers().Tadj / (tauh * 2);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);

    g[i] = gbar[i] * m[i] * (frac_inact * h[i] + (1-frac_inact));
#elif CHANNEL_KIR == KIR_MAHON_2000
    //dyn_var_t qm = dt * getSharedMembers().Tadj / (TAUM * 2);
    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    m[i] = m_inf;
    g[i] = gbar[i] * m[i];
#else
    NOT IMPLEMENTED YET;
#endif
    { // trick to keep m in [0, 1]
      if (m[i] < 0.0) { m[i] = 0.0; }
      else if (m[i] > 1.0) { m[i] = 1.0; }
    }
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]); // at time (t+dt/2)
  }
}

// GOAL: To meet second-order derivative, the gates is calculated to
//     give the value at time (t0+dt/2) using data voltage v(t0)
//  NOTE:
//    If steady-state formula is used, then the calculated value of gates
//            is at time (t0); but as steady-state, value at time (t0+dt/2) is the same
//    If non-steady-state formula (dy/dt = f(v)) is used, then
//        once gate(t0) is calculated using v(t0)
//        we need to estimate gate(t0+dt/2)
//                  gate(t0+dt/2) = gate(t0) + f(v(t0)) * dt/2
void ChannelKIR::initialize(RNG& rng)
{
#if CHANNEL_KIR == KIR2_1_STEEPHEN_MANCHANDA_2009 || \
    CHANNEL_KIR == KIR2_1_TUAN_JAMES_2017
  if (frac_inact < SMALL)
    frac_inact = 0.47; //default fraction of channel goes through inactivation
#endif
  pthread_once(&once_KIR, ChannelKIR::initialize_others);
  assert(branchData);
  unsigned size = branchData->size;
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (m.size() != size) m.increaseSizeTo(size);
#if CHANNEL_KIR == KIR2_1_STEEPHEN_MANCHANDA_2009 || \
    CHANNEL_KIR == KIR2_1_TUAN_JAMES_2017
  if (h.size() != size) h.increaseSizeTo(size);
#endif
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
  // initialize
  float gbar_default = gbar[0];
  if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels KIR Param"
      << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
    if (gbar_dists.size() > 0) {
      unsigned int j;
      //NOTE: 'n' bins are splitted by (n-1) points
      if (gbar_values.size() - 1 != gbar_dists.size())
      {
        std::cerr << "gbar_values.size = " << gbar_values.size()
          << "; gbar_dists.size = " << gbar_dists.size() << std::endl;
      }
      assert(gbar_values.size() -1 == gbar_dists.size());
      for (j=0; j<gbar_dists.size(); ++j) {
        if ((*dimensions)[i]->dist2soma < gbar_dists[j]) break;
      }
      gbar[i] = gbar_values[j];
    }
    /*else if (gbar_values.size() == 1) {
      gbar[i] = gbar_values[0];
    } */
    else if (gbar_branchorders.size() > 0)
    {
      unsigned int j;
      assert(gbar_values.size() == gbar_branchorders.size());
      SegmentDescriptor segmentDescriptor;
      for (j=0; j<gbar_branchorders.size(); ++j) {
        if (segmentDescriptor.getBranchOrder(branchData->key) == gbar_branchorders[j]) break;
      }
      if (j == gbar_branchorders.size() and gbar_branchorders[j-1] == GlobalNTS::anybranch_at_end)
      {
	gbar[i] = gbar_values[j-1];
      }
      else if (j < gbar_values.size())
        gbar[i] = gbar_values[j];
      else
        gbar[i] = gbar_default;
    }
    else {
      gbar[i] = gbar_default;
    }
  }
  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_KIR == KIR_HAYASHI_FISHMAN_1988 || \
    CHANNEL_KIR == KIR_WOLF_2005 || \
    CHANNEL_KIR == KIR_STEEPHEN_MANCHANDA_2009
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    g[i] = gbar[i] * m[i];
#elif CHANNEL_KIR == KIR_MAHON_2000
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    g[i] = gbar[i] * m[i];

#elif CHANNEL_KIR == KIR2_1_STEEPHEN_MANCHANDA_2009 || \
      CHANNEL_KIR == KIR2_1_TUAN_JAMES_2017
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_tauh.begin(), Vmrange_tauh.end(), v);
    int index = low - Vmrange_tauh.begin();
    float h_inf;
    if (index == 0)
      h_inf = h_inf_KIR[0];
    else
      h_inf = linear_interp(Vmrange_tauh[index-1], h_inf_KIR[index-1],
        Vmrange_tauh[index], h_inf_KIR[index], v);
    h[i] = h_inf;

    g[i] = gbar[i] * m[i] * (frac_inact * h_inf + (1-frac_inact));
#else
    NOT IMPLEMENTED YET;
// m[i] = am / (am + bm); //steady-state value
#endif
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
  }
}

void ChannelKIR::initialize_others()
{
#if CHANNEL_KIR == KIR_HAYASHI_FISHMAN_1988 || \
    CHANNEL_KIR == KIR_WOLF_2005 || \
    CHANNEL_KIR == KIR_STEEPHEN_MANCHANDA_2009 || \
    CHANNEL_KIR == KIR2_1_STEEPHEN_MANCHANDA_2009 || \
    CHANNEL_KIR == KIR2_1_TUAN_JAMES_2017
  {
    std::vector<dyn_var_t> tmp(_Vmrange_taum, _Vmrange_taum + LOOKUP_TAUM_LENGTH);
    assert(sizeof(taumKIR) / sizeof(taumKIR[0]) == tmp.size());
    //Vmrange_taum.resize(tmp.size()-2);
    //for (unsigned long i = 1; i < tmp.size() - 1; i++)
    //  Vmrange_taum[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
    Vmrange_taum = tmp;
  }
#if CHANNEL_KIR == KIR2_1_STEEPHEN_MANCHANDA_2009 || \
    CHANNEL_KIR == KIR2_1_TUAN_JAMES_2017
  {
    std::vector<dyn_var_t> tmp(_Vmrange_tauh,
                               _Vmrange_tauh + LOOKUP_TAUH_LENGTH);
    assert(sizeof(tauhKIR) / sizeof(tauhKIR[0]) == tmp.size());
    //Vmrange_tauh.resize(tmp.size()-2);
    //for (unsigned long i = 1; i < tmp.size() - 1; i++)
    //  Vmrange_tauh[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
    Vmrange_tauh = tmp;
  }
#endif
#endif
}

ChannelKIR::~ChannelKIR() {}
