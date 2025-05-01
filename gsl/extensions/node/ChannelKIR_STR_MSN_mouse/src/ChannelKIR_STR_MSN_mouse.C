// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#include "Mgs.h"
#include "ChannelKIR_STR_MSN_mouse.h"
#include "CG_ChannelKIR_STR_MSN_mouse.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"
#include "NumberUtils.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>
static pthread_once_t once_KIR = PTHREAD_ONCE_INIT;

// Mahon 2000:
//assume activation is instantaneous
#define VHALF_M -100 
#define k_M 10 
//#define TAUM 1 //ms                  

// GOAL: update gates using v(t+dt/2) and gate(t-dt/2)
//   --> output gate(t+dt/2)
//   of second-order accuracy at time (t+dt/2) using trapezoidal rule
void ChannelKIR_STR_MSN_mouse::update(RNG& rng) 
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
    //dyn_var_t qm = dt * getSharedMembers().Tadj / (TAUM * 2);
    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M)); 
    m[i] = m_inf;
    g[i] = gbar[i] * m[i];
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
void ChannelKIR_STR_MSN_mouse::initialize(RNG& rng) 
{
  assert(branchData);
  unsigned size = branchData->size;
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (m.size() != size) m.increaseSizeTo(size);
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
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    g[i] = gbar[i] * m[i];
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
  }
}

ChannelKIR_STR_MSN_mouse::~ChannelKIR_STR_MSN_mouse() 
{
}

