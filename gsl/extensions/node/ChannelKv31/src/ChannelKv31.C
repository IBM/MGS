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
*/


#include "Lens.h"
#include "ChannelKv31.h"
#include "CG_ChannelKv31.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "GlobalNTSConfig.h"
#include "NumberUtils.h"
#include "SegmentDescriptor.h"
#include "Branch.h"
#include <math.h>
#include <pthread.h>
#include <algorithm>

#define SMALL 1.0E-6
#define fieldDelimiter "\t"

static pthread_once_t once_Nat = PTHREAD_ONCE_INIT;
#if defined(WRITE_GATES)                                      
SegmentDescriptor ChannelKv31::_segmentDescriptor;
#endif


#if CHANNEL_Kv31 == Kv31_RETTIG_1992
#define IMV -18.7
#define IMD 9.7
#define TMF 4.0
#define TMV 46.56
#define TMD 44.14
#define T_ADJ 2.9529 // 2.3^((34-21)/10)

#elif CHANNEL_Kv31 == Kv31_FUJITA_2012

// Gate: m^4 * h                                                            
//Reference from Fujita et al, Kv3 fast delayed rectifier
//modification of GPe neuron model proposed by Gunay et al (GPe neuron in basal ganglia)                 
//                                                      
//m activation, h inactivation,
// dm/dt = (minf( V ) - V)/tau_m(V) 
// dh/dt = (hinf( V ) - V)/tau_h(V) 
// minf  = 1 / (1 + exp( (V - VHALF_M) / k_M))                               
// hinf  = H_MIN + (1-H_MIN) / (1 + exp( (V - VHALF_H) / k_H))                               
// tau_m = TAU0_M + (TAU1_M - TAU0_M) / ( exp( (PHI_M - V)/SIG0_M) + exp( (PHI_M - V)/SIG1_M) )
 
// tau_h = TAU0_H + (TAU1_H - TAU0_H) / ( exp( (PHI_H - V)/SIG0_H) + exp( (PHI_H - V)/SIG1_H) )
//
// NOTE: vtrap(x,y) = x/(exp(x/y)-1)                                                
#define VHALF_M -26

#define k_M 7.8

#define VHALF_H -20

#define k_H -10

#define H_MIN 0.6

#define TAU0_M 0.1

#define TAU1_M 14

#define PHI_M -26

#define SIG0_M 13

#define SIG1_M -12

#define TAU0_H 7.0

#define TAU1_H 33

#define PHI_H 0.0

#define SIG0_H 10

#define SIG1_H -10




#else
  NOT DEFINED YET

#endif


void ChannelKv31::update(RNG& rng) 
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
#if CHANNEL_Kv31 == Kv31_RETTIG_1992
    dyn_var_t minf = 1.0/(1.0 + exp(-(v + IMV)/IMD));
    // dyn_var_taum = TMF/(T_ADJ*(1 + exp(-(v + TMV)/TMD)));
    dyn_var_t taum = TMF/((1 + exp(-(v + TMV)/TMD)) * getSharedMembers().Tadj);
    dyn_var_t pm = 0.5*dt/taum;
    m[i] = (2.0*pm*minf + m[i]*(1.0 - pm))/(1.0 + pm);
 #elif CHANNEL_Kv31 == Kv31_FUJITA_2012
    {
 dyn_var_t taum = TAU0_M + (TAU1_M - TAU0_M) / ( exp( (PHI_M - v)/SIG0_M) + exp( (PHI_M - v)/SIG1_M) );
   dyn_var_t qm = dt  / (taum * 2.0);
  dyn_var_t tauh = TAU0_H + (TAU1_H - TAU0_H) / ( exp( (PHI_H - v)/SIG0_H) + exp( (PHI_H - v)/SIG1_H) );
   dyn_var_t qh = dt  / (tauh * 2.0);
   dyn_var_t m_inf = 1.0 / (1.0 + exp(( VHALF_M - v) / k_M));
   dyn_var_t h_inf  = H_MIN + (1.0-H_MIN) / (1.0 + exp( ( VHALF_H - v) / k_H));                               
    m[i] = (2.0 * m_inf * qm - m[i] * (qm - 1.0)) / (qm + 1.0);
    h[i] = (2.0 * h_inf * qh - h[i] * (qh - 1.0)) / (qh + 1.0);
    } 
#else
 NOT IMPLEMENTED YET;    
#endif
#if defined(WRITE_GATES)                                                  
    if (is_write)
    {           
      (*outFile) << std::fixed << fieldDelimiter << m[i];                 
 #if CHANNEL_Kv31 == Kv31_FUJITA_2012
      (*outFile) << std::fixed << fieldDelimiter << h[i];
 #endif
   } 
#endif
 
    // trick to keep m in [0, 1]
    if (m[i] < 0.0) { m[i] = 0.0; }
    else if (m[i] > 1.0) { m[i] = 1.0; }
#if CHANNEL_Kv31 == Kv31_FUJITA_2012 
    if (h[i] < 0.0) { h[i] = 0.0; }
    else if (h[i] > 1.0) { h[i] = 1.0; }
#endif
#if CHANNEL_Kv31 == Kv31_RETTIG_1992
    g[i] = gbar[i] * m[i] ;
#elif CHANNEL_Kv31 == Kv31_FUJITA_2012
    g[i] = gbar[i] * m[i] * m[i] * m[i] * m[i] * h[i];
#endif
		Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
  }
}

void ChannelKv31::initialize(RNG& rng) 
{
  //pthread_once(&once_KAf, ChannelKAf::initialize_others);
  assert(branchData);
  unsigned size = branchData->size;
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (m.size() != size) m.increaseSizeTo(size);
#if CHANNEL_Kv31 == Kv31_FUJITA_2012
  if (h.size() != size) h.increaseSizeTo(size);
#endif 
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
  // initialize
  float gbar_default = gbar[0];
	if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
	{
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels Kv31 Param"
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
      if (gbar_values.size() != gbar_branchorders.size())
      {
        std::cerr << "gbar_values.size = " << gbar_values.size()
          << "; gbar_branchorders.size = " << gbar_branchorders.size() << std::endl;
      }
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
#if CHANNEL_Kv31 == Kv31_RETTIG_1992
    m[i] = 1.0/(1.0 + exp(-(v + IMV)/IMD));
    g[i] = gbar[i]*m[i];
#elif CHANNEL_Kv31 == Kv31_FUJITA_2012
   m[i] = 1.0 / (1 + exp((VHALF_M-v) / k_M));
   h[i]  = H_MIN + (1-H_MIN) / (1 + exp( (VHALF_H-v) / k_H));       
   g[i] = gbar[i] * m[i] * m[i] * m[i] * m[i]*  h[i];
#else
    NOT IMPLEMENTED YET;
// m[i] = am / (am + bm);  // steady-state value
// h[i] = ah / (ah + bh);
#endif
#if defined(WRITE_GATES)                                      
    if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
        _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
    {
      (*outFile) << std::fixed << fieldDelimiter << m[i];       
#if CHANNEL_Kv31 == Kv31_FUJITA_2012
      (*outFile) << std::fixed << fieldDelimiter << h[i];  
#endif
   } 
#endif                                                        
 

		Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
 }
}

ChannelKv31::~ChannelKv31() 
{
}

