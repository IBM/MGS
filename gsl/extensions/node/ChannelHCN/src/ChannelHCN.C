// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "ChannelHCN.h"
#include "CG_ChannelHCN.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"

#include <math.h>
#include <pthread.h>
#include <algorithm>

#define SMALL 1.0E-6

#if CHANNEL_HCN == HCN_HUGUENARD_MCCORMICK_1992
/*
 * Huguenard-McCormick (1992) Journal of neurophysiology
 *  Simulation of currents ... rhythmic oscillations of Thalamic Relay neurons
 * McCormick-Huguenard (1992) Journal of neurophysiology
 *  Model of electrophysiological properties of thalamic relay neurons
I = ghat * gmax * (V-Erev)
ghat = m^N * h
As the h-current does not inactivated (even with prolonged hyperpolarization in
minutes)
so 'h' is always set to 1
The time constant of activation and inactivation is fitted using single
exponential
The rate of activation and inactivation 'tau_m' is modeled using bell-shaped
tau_m = 1. / (exp (-14.59 - 0.086 Vm)+ exp(-1.87 + 0.0701 Vm))
Erev = -43 mV
gmax = 15 to 30 nS
NOTE:
The choice N = 1 is justified as there is no apparent delay in activation
 */

#define VHALF_M -75.0  //[mV]
#define k_M 5.5        // [mV^{-1}]
#define POW_M 1

#elif CHANNEL_HCN == HCN_VANDERGIESSEN_DEZEEUW_2008
/*  Van Der Giessen ... De Zeeuw (2008) Neuron
 *   Role of Olivary Electrical coupling in cerebellar motor learning
 *  NOTE: The h-current was moved to dendritic compartment and thus was modified
 *     based on Huguenard-McCormick (1992)
 */

#define VHALF_M -80.0  //[mV]
#define k_M 4.0        // [mV^{-1}]
#define POW_M 1

#elif CHANNEL_HCN == HCN_KOLE_2006 || \
	  CHANNEL_HCN == HCN_HAY_2011
/* Kole et al. (2006) 
// Simulation suggests: gh ~ 2.3 pS/um^2 at soma
//                         ~  93 pS/um^2 distal apical dendrites ~ 1000 um from soma
//  Erev_h = -45 mV
//
// a_m  = AMC*(V + AMV)/( exp( (V + AMV)/AMD ) - 1.0 )
// b_m  = BMC * exp( (V + BMV)/BMD )
// a_h  = AHC * exp( (V + AHV)/AHD )
// b_h  = BHC / (exp( (V + BHV)/BHD ) + 1.0)
*/
#define AMC 0.00643
#define AMV 154.9
#define AMD 11.9
#define BMC 0.193
#define BMD 33.1
#endif

dyn_var_t ChannelHCN::conductance(int i)
{
	//conductance density gh is distributed across compartments
	//using the exponential functions: gh = y0 + A * exp(d/lambda)
	//with y0 = -2 pS/um^2
	//     A  = 4.28 pS/um^2
	//     lambda = 323 um
	//     d  = distance from the soma
	float d = (*dimensions)[i]->dist2soma;
	const float y0 = -2.0 ;
	const float A = 4.28 ; 
	const float	lambda = 323.0 ;
	dyn_var_t gh = y0 + A * exp(d/lambda);
	return gh;
}



// NOTE: vtrap(x,y) = x/(exp(x/y)-1)
dyn_var_t ChannelHCN::vtrap(dyn_var_t x, dyn_var_t y)
{
  return (fabs(x / y) < SMALL ? y * (1 - x / y / 2) : x / (exp(x / y) - 1));
}

void ChannelHCN::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_HCN == HCN_HUGUENARD_MCCORMICK_1992
    // NOTE: Some models use m_inf and tau_m to estimate m
	dyn_var_t taum = 1.0 / ( exp(-0.086 * v - 14.6) + exp (0.07 * v - 1.87) );
    dyn_var_t qm = dt * getSharedMembers().Tadj / (taum * 2);
    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    // see Rempe-Chomp (2006)
    m[i] = (2.0 * m_inf * qm - m[i] * (qm - 1.0)) / (qm + 1.0);
#elif CHANNEL_HCN == HCN_KOLE_2006
	//to be implemented (probably put into a new _Markov channel name for stochastic simulation)
	assert(0);
#elif CHANNEL_HCN == HCN_VANDERGIESSEN_DEZEEUW_2008
    // NOTE: Some models use m_inf and tau_m to estimate m
	dyn_var_t taum = 1.0 / ( exp(-0.086 * v - 14.6) + exp (0.07 * v - 1.87) );
    dyn_var_t qm = dt * getSharedMembers().Tadj / (taum * 2);
    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    // see Rempe-Chomp (2006)
    m[i] = (2.0 * m_inf * qm - m[i] * (qm - 1.0)) / (qm + 1.0);
#elif CHANNEL_HCN == HCN_HAY_2011
    // NOTE: Some models use alpha_m and beta_m to estimate m
    dyn_var_t am = AMC * vtrap(v + AMV, AMD);
    dyn_var_t bm = BMC * exp(v / BMD);
    //m_infty = am / (am+bm)
    //tau_m = 1/(am+bm)
    // see Rempe-Chomp (2006)
    dyn_var_t pm = 0.5 * dt * (am + bm) * getSharedMembers().Tadj;
    m[i] = (dt * am * getSharedMembers().Tadj + m[i] * (1.0 - pm)) / (1.0 + pm);
    g[i] = gbar[i] * m[i];
#else
	assert(0);
#endif
		Iion[i] = g[i] * (v - getSharedMembers().E_HCN[0]);
  }
}

void ChannelHCN::initialize(RNG& rng)
{
  unsigned size = branchData->size;
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);

  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (m.size() != size) m.increaseSizeTo(size);
  if (Iion.size()!=size) Iion.increaseSizeTo(size);

  // initialize
  // NOTE: add the scaling_factor for testing channel easier
  //  Should be set to 1.0 by default
  dyn_var_t scaling_factor = 1.0;
  float gbar_default = gbar[0] * scaling_factor;
  //float gbar_default = gbar[0];
  if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
  {
    std::cerr
        << "ERROR: Use either gbar_dists or gbar_branchorders on Channels HCN Param"
        << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
    if (gbar_dists.size() > 0)
    {
      unsigned int j;
	  //NOTE: 'n' bins are splitted by (n-1) points
	  //gbar_dists = hold such points
	  //gbar_values = hold value in each bin
      if (gbar_values.size() - 1 != gbar_dists.size())
      {
        std::cerr << "gbar_values.size = " << gbar_values.size() 
          << "; gbar_dists.size = " << gbar_dists.size() << std::endl; 
      }
      assert(gbar_values.size() -1 == gbar_dists.size());
      for (j = 0; j < gbar_dists.size(); ++j)
      {
        if ((*dimensions)[i]->dist2soma < gbar_dists[j]) break;
      }
      gbar[i] = gbar_values[j] * scaling_factor;
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
        gbar[i] = gbar_values[j - 1] * scaling_factor;
      }
      else if (j < gbar_values.size())
        gbar[i] = gbar_values[j] * scaling_factor;
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
#if CHANNEL_HCN == HCN_HUGUENARD_MCCORMICK_1992
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    g[i] = gbar[i] * m[i];  // pow(m[i], POW_M)
#elif CHANNEL_HCN == HCN_VANDERGIESSEN_DEZEEUW_2008
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    g[i] = gbar[i] * m[i];  // pow(m[i], POW_M)
#elif CHANNEL_HCN == HCN_KOLE_2006 
    gbar[i] = conductance(i);
    NumChan[i] = dimensions[i].surface_area * ChanDen;
#elif CHANNEL_HCN == HCN_HAY_2011
   //NumChan[i] = 1
	//gbar[i] = conductance(i);
    dyn_var_t am = AMC * vtrap(v + AMV, AMD);
    dyn_var_t bm = BMC * exp(v / BMD);
    m[i] = am / (am + bm);
    g[i] = gbar[i] * m[i];
#else
    assert(0);
#endif
		Iion[i] = g[i] * (v - getSharedMembers().E_HCN[0]);
  }
}

ChannelHCN::~ChannelHCN() {}
