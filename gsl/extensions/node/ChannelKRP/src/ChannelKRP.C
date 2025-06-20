// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "CG_ChannelKRP.h"
#include "ChannelKRP.h"
#include "Mgs.h"
#include "rndm.h"

#include "GlobalNTSConfig.h"
#include "NumberUtils.h"
#include "SegmentDescriptor.h"

#define SMALL 1.0E-6
#include <algorithm>
#include <math.h>
#include <pthread.h>
static pthread_once_t once_KRP = PTHREAD_ONCE_INIT;

//
// This is an implementation of the "4-AP resistant persistent
//                  KRP potassium current
// 4-AP-(r)esistant, (p)ersistent K+ current: KRP or Krp
//
#if CHANNEL_KRP == KRP_WOLF_2005
// Inactivation from
//   1. Nisenbaum et al. (1996), Fig. 9D = V1/2, slope; Fig. 9A = (35^C)
//   fraction inactivation
// Activation from
//   1. Nisenbaum et al. (1996), Fig. 6C
// Model: partial inactivation
// minf(Vm) = 1/(1+exp((Vm-Vh)/k))
// hinf(Vm) = 1/(1+exp(Vm-Vh)/k)
#define VHALF_M -13.5
#define k_M -11.8  // NOTE: The Nisenbaum paper is 11.8
#define VHALF_H -54.7
#define k_H 18.6
#define frac_inact 0.7  // 'a' term
#define LOOKUP_TAUM_LENGTH 31
const dyn_var_t ChannelKRP::_Vmrange_taum[] = {
    -100, -95, -90, -85, -80, -75, -70, -65, -60, -55, -50,
    -45,  -40, -35, -30, -25, -20, -15, -10, -5,  0,   5,
    10,   15,  20,  25,  30,  35,  40,  45,  50};
dyn_var_t ChannelKRP::taumKRP[] = {40,   45, 48.8, 55, 64.4, 75, 83.9, 90,
                                   93.5, 95, 95.4, 97, 99.2, 95, 79.7, 60,
                                   44.5, 35, 29.3, 25, 20,   15, 11.6, 10,
                                   9.6,  10, 10.5, 10, 8,    5,  5};
#define LOOKUP_TAUH_LENGTH 31
const dyn_var_t ChannelKRP::_Vmrange_tauh[] = {
    -100, -95, -90, -85, -80, -75, -70, -65, -60, -55, -50,
    -45,  -40, -35, -30, -25, -20, -15, -10, -5,  0,   5,
    10,   15,  20,  25,  30,  35,  40,  45,  50};
dyn_var_t ChannelKRP::tauhKRP[] = {
    7000.0, 7000.0, 7000.0, 7000.0, 7000.0, 7000.0, 7000.0, 7000.0,
    7000.0, 7000.0, 7000.0, 7000.0, 7000.0, 7000.0, 7000.0, 7000.0,
    7000.0, 7000.0, 7000.0, 7000.0, 6742.5, 6000.0, 4740.2, 3500.0,
    2783.3, 2500.0, 2336.3, 2200.0, 2083.5, 2000.0, 2000.0};
std::vector<dyn_var_t> ChannelKRP::Vmrange_taum;
std::vector<dyn_var_t> ChannelKRP::Vmrange_tauh;
#elif CHANNEL_KRP == KRP_MAHON_2000
#define VHALF_M -13.4
#define k_M -12.1
#define VHALF_H -55.0
#define k_H 19.0

#define tau_M 206.2
#define VHALF_TAUM -53.9
#define k_TAUM 26.5

#define VHALF_TAUH -38.2
#define k_TAUH 28
#else
NOT IMPLEMENTED YET
// const dyn_var_t _Vmrange_taum[] = {};
// const dyn_var_t _Vmrange_tauh[] = {};
// const dyn_var_t taumKRP[] = {};
// const dyn_var_t tauhKRP[] = {};
#endif

// GOAL: update gates using v(t+dt/2) and gate(t-dt/2)
//   --> output gate(t+dt/2)
//   of second-order accuracy at time (t+dt/2) using trapezoidal rule
void ChannelKRP::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_KRP == KRP_WOLF_2005
    // NOTE: Some models use m_inf and tau_m to estimate m
    // tau_m in the lookup table
    std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
    int index = low - Vmrange_taum.begin();
    //-->tau_m[i] = taumKRP[index];
    // NOTE: dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m[i] * 2);
    // dyn_var_t qm = dt * getSharedMembers().Tadj / (taumKRP[index] * 2);
    dyn_var_t taum;
    if (index == 0)
      taum = taumKRP[0];
    else
      taum = linear_interp(Vmrange_taum[index - 1], taumKRP[index - 1],
                           Vmrange_taum[index], taumKRP[index], v);
    dyn_var_t qm = dt * getSharedMembers().Tadj / (taum * 2);
    /* no need to search as they both use the same Vmrange
     * IF NOT< make sure you add this code
    std::vector<dyn_var_t>::iterator low= std::lower_bound(Vmrange_tauh.begin(),
    Vmrange_tauh.end(), v);
    int index = low-Vmrange_tauh.begin();
    */
    // dyn_var_t qh = dt * getSharedMembers().Tadj / (tauhKRP[index] * 2);
    dyn_var_t tauh;
    if (index == 0)
      tauh = tauhKRP[0];
    else
      tauh = linear_interp(Vmrange_tauh[index - 1], tauhKRP[index - 1],
                           Vmrange_tauh[index], tauhKRP[index], v);
    dyn_var_t qh = dt * getSharedMembers().Tadj / (tauh * 2);

    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
#elif CHANNEL_KRP == KRP_MAHON_2000
    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));

    dyn_var_t tau_m = tau_M / (exp(-(v - VHALF_TAUM) / k_TAUM) +
                               exp((v - VHALF_TAUM) / k_TAUM)) /
      getSharedMembers().Tadj;
    dyn_var_t tau_h = 3 *
                      (1790 + 2930 * exp(-pow((v - VHALF_TAUH) / k_TAUH, 2)) *
                      ((v - VHALF_TAUH) / k_TAUH)) /
                      getSharedMembers().Tadj;

    dyn_var_t qm = dt / (tau_m * 2);
    dyn_var_t qh = dt / (tau_h * 2);

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
#else
    NOT IMPLEMENTED YET
#endif
    {
      // trick to keep m in [0, 1]
      if (m[i] < 0.0)
      {
        m[i] = 0.0;
      }
      else if (m[i] > 1.0)
      {
        m[i] = 1.0;
      }
      // trick to keep m in [0, 1]
      if (h[i] < 0.0)
      {
        h[i] = 0.0;
      }
      else if (h[i] > 1.0)
      {
        h[i] = 1.0;
      }
    }

#if CHANNEL_KRP == KRP_WOLF_2005
    g[i] = gbar[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
#elif CHANNEL_KRP == KRP_MAHON_2000
    g[i] = gbar[i] * m[i] * h[i];
#endif
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);  // at time (t+dt/2)
  }
}

// GOAL: To meet second-order derivative, the gates is calculated to
//     give the value at time (t0+dt/2) using data voltage v(t0)
//  NOTE:
//    If steady-state formula is used, then the calculated value of gates
//            is at time (t0); but as steady-state, value at time (t0+dt/2) is
//            the same
//    If non-steady-state formula (dy/dt = f(v)) is used, then
//        once gate(t0) is calculated using v(t0)
//        we need to estimate gate(t0+dt/2)
//                  gate(t0+dt/2) = gate(t0) + f(v(t0)) * dt/2
void ChannelKRP::initialize(RNG& rng)
{
  pthread_once(&once_KRP, ChannelKRP::initialize_others);
  assert(branchData);
  unsigned size = branchData->size;
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (m.size() != size) m.increaseSizeTo(size);
  if (h.size() != size) h.increaseSizeTo(size);
  if (Iion.size() != size) Iion.increaseSizeTo(size);
  // initialize
  float gbar_default = gbar[0];
  if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
  {
    std::cerr
        << "ERROR: Use either gbar_dists or gbar_branchorders on Channels Param"
        << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
    if (gbar_dists.size() > 0)
    {
      unsigned int j;
      // NOTE: 'n' bins are splitted by (n-1) points
      if (gbar_values.size() - 1 != gbar_dists.size())
      {
        std::cerr << "gbar_values.size = " << gbar_values.size()
                  << "; gbar_dists.size = " << gbar_dists.size() << std::endl;
      }
      assert(gbar_values.size() - 1 == gbar_dists.size());
      for (j = 0; j < gbar_dists.size(); ++j)
      {
        if ((*dimensions)[i]->dist2soma < gbar_dists[j]) break;
      }
      gbar[i] = gbar_values[j];
    }
    else if (gbar_branchorders.size() > 0)
    {
      unsigned int j;
      assert(gbar_values.size() == gbar_branchorders.size());
      SegmentDescriptor segmentDescriptor;
      for (j = 0; j < gbar_branchorders.size(); ++j)
      {
        if (segmentDescriptor.getBranchOrder(branchData->key) ==
            gbar_branchorders[j])
          break;
      }
      if (j == gbar_branchorders.size() and
          gbar_branchorders[j - 1] == GlobalNTS::anybranch_at_end)
      {
        gbar[i] = gbar_values[j - 1];
      }
      else if (j < gbar_values.size())
        gbar[i] = gbar_values[j];
      else
        gbar[i] = gbar_default;
    }
    else
    {
      gbar[i] = gbar_default;
    }
  }
  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_KRP == KRP_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
    g[i] = gbar[i] * m[i] * (frac_inact * h[i] + (1 - frac_inact));
#elif CHANNEL_KRP == KRP_MAHON_2000
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
    g[i] = gbar[i] * m[i] * h[i];
#else
    NOT IMPLEMENTED YET;
#endif
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);  // at time (t0+dt/2)
  }
}

void ChannelKRP::initialize_others()
{
#if CHANNEL_KRP == KRP_WOLF_2005
  {
    std::vector<dyn_var_t> tmp(_Vmrange_taum,
                               _Vmrange_taum + LOOKUP_TAUM_LENGTH);
    assert((sizeof(taumKRP) / sizeof(taumKRP[0])) == tmp.size());
    // Vmrange_taum.resize(tmp.size()-2);
    // for (unsigned long i = 1; i < tmp.size() - 1; i++)
    //  Vmrange_taum[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
    Vmrange_taum = tmp;
  }
  {
    std::vector<dyn_var_t> tmp(_Vmrange_tauh,
                               _Vmrange_tauh + LOOKUP_TAUH_LENGTH);
    assert(sizeof(tauhKRP) / sizeof(tauhKRP[0]) == tmp.size());
    // Vmrange_tauh.resize(tmp.size()-2);
    // for (unsigned long i = 1; i < tmp.size() - 1; i++)
    //  Vmrange_tauh[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
    Vmrange_tauh = tmp;
  }
#endif
}

ChannelKRP::~ChannelKRP() {}
