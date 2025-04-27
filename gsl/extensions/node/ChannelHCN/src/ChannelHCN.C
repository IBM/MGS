// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
// 
// =============================================================================
// 
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
// 
// =============================================================================

#include "Lens.h"
#include "ChannelHCN.h"
#include "CG_ChannelHCN.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"
#include "NumberUtils.h"
#include "MaxComputeOrder.h"
#include "Branch.h"

#include <math.h>
#include <pthread.h>
#include <algorithm>
#include <cmath>

#define SMALL 1.0E-6
#define decimal_places 6     
#define fieldDelimiter "\t"  

#if defined(WRITE_GATES)                                      
SegmentDescriptor ChannelHCN::_segmentDescriptor;
#endif


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

#elif CHANNEL_HCN == HCN_FUJITA_2012                                                
// Gate: m                                                               
//Reference from Fujita et al, 
// Hyperpolarization-activated cyclic nucleotide-modulated cation current
//modification of GPe neuron model proposed by Gunay et al (GPe neuron in basal ganglia)                 
//                                                      
//m activation
// dm/dt = (minf( V ) - V)/tau_m(V) 
//  dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M - Vhalf_m_shift) / k_M));
// minf  = 1 / (1 + exp( (V - VHALF_M) / k_M))                               
// tau_h = TAU0_M + (TAU1_M - TAU0_M) / ( exp( (PHI_M - V)/SIG0_M) + exp( (PHI_M - V)/SIG1_M) )
// NOTE: vtrap(x,y) = x/(exp(x/y)-1)                                                
//#define Vshift 7  // [mV]                                                           
#define VHALF_M -76.4

#define k_M -3.3

#define TAU0_M 0

#define TAU1_M 3625

#define PHI_M -76.4

#define SIG0_M 6.56

#define SIG1_M -7.48

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




void ChannelHCN::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);

#if defined(WRITE_GATES)                                                  
  bool is_write = false;
  if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
      _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
  {
    float currentTime = float(getSimulation().getIteration()) * dt + dt/2;       
    if (currentTime >= _prevTime + IO_INTERVAL)                           
    {                                                                     
      (*outFile) << std::endl;                                            
      (*outFile) <<  currentTime;                                         
      _prevTime = currentTime;                                            
      is_write = true;
    }
  }
#endif
 
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
#elif CHANNEL_HCN == HCN_FUJITA_2012
     {
   dyn_var_t taum = TAU0_M + (TAU1_M - TAU0_M) / ( exp( (PHI_M - v)/SIG0_M) + exp( (PHI_M - v)/SIG1_M) );
   dyn_var_t qm = dt  / (taum * 2);
   dyn_var_t m_inf = 1.0 / (1 + exp(( VHALF_M - v) / k_M));
    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    g[i] = gbar[i] * m[i];
    }


#else
	assert(0);
#endif
		Iion[i] = g[i] * (v - getSharedMembers().E_HCN[0]);
 #if defined(WRITE_GATES)                                                  
    if (is_write)
    {           
      (*outFile) << std::fixed << fieldDelimiter << m[i];                 
   }                                                                     
#endif                                                                    
 
  }
}

void ChannelHCN::initialize(RNG& rng)
{
    if (std::abs(gbar_scale) <= 0.000001)
        gbar_scale = 1.0;
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
  //dyn_var_t scaling_factor = 1.0;
  //float gbar_default = gbar[0] * scaling_factor;
  float gbar_default = gbar[0] ;
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
      gbar[i] = gbar_values[j] ;
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
        gbar[i] = gbar_values[j - 1] ;
      }
      else if (j < gbar_values.size())
        gbar[i] = gbar_values[j] ;
      else
        gbar[i] = gbar_default;
    }
    else
    {
      gbar[i] = gbar_default;
    }

    gbar[i] = gbar[i] * gbar_scale;
  }

#if defined(WRITE_GATES)                                      
  if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
      _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
  {
    std::ostringstream os;                                    
    std::string fileName = "gates_HCN.txt";                       
    os << fileName << getSimulation().getRank() ;              
    outFile = new std::ofstream(os.str().c_str());            
    outFile->precision(decimal_places);    
(*outFile) << "#Time" << fieldDelimiter << "gates: m[, m]*"; 
    _prevTime = 0.0;                                          
    float currentTime = 0.0;  // should also be (dt/2)                                 
    (*outFile) << std::endl;                                  
    (*outFile) <<  currentTime;                               
  }
#endif
 

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
#elif CHANNEL_NAT == NAT_FUJITA_2012
   m[i] = 1.0 / (1 + exp((VHALF_M-v) / k_M)); 
   g[i] = gbar[i] * m[i];
#else
    assert(0);
#endif
		Iion[i] = g[i] * (v - getSharedMembers().E_HCN[0]);

#if defined(WRITE_GATES)                                      
    if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
        _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
    {
      (*outFile) << std::fixed << fieldDelimiter << m[i];       
    }
#endif                                                        
 
  }
}

ChannelHCN::~ChannelHCN() {}
