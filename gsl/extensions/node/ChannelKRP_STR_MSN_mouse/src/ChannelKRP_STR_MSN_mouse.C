// =================================================================
//
// (C) Copyright 2018 New Jersey Institute of Technology.
//
// =================================================================


#include "Lens.h"
#include "ChannelKRP_STR_MSN_mouse.h"
#include "CG_ChannelKRP_STR_MSN_mouse.h"
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
// Mahon 2000:
#define VHALF_M -13.4
#define k_M -12.1
#define VHALF_H -55.0
#define k_H 19.0

#define tau_M 206.2
#define VHALF_TAUM -53.9
#define k_TAUM 26.5

#define VHALF_TAUH -38.2
#define k_TAUH 28

// GOAL: update gates using v(t+dt/2) and gate(t-dt/2)
//   --> output gate(t+dt/2)
//   of second-order accuracy at time (t+dt/2) using trapezoidal rule
void ChannelKRP_STR_MSN_mouse::update(RNG& rng) 
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
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

    g[i] = gbar[i] * m[i] * h[i];
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
void ChannelKRP_STR_MSN_mouse::initialize(RNG& rng) 
{
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
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
    g[i] = gbar[i] * m[i] * h[i];
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);  // at time (t0+dt/2)
  }
}

ChannelKRP_STR_MSN_mouse::~ChannelKRP_STR_MSN_mouse() 
{
}

