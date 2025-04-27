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
#include "ChannelMK.h"
#include "CG_ChannelMK.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "MaxComputeOrder.h"
#include "GlobalNTSConfig.h"
#include "NumberUtils.h"
#include <pthread.h>

#define SMALL 1.0E-6
#define decimal_places 6     
#define fieldDelimiter "\t"  

static pthread_once_t once_MK = PTHREAD_ONCE_INIT;
#if defined(WRITE_GATES)                                      
SegmentDescriptor ChannelMK::_segmentDescriptor;
#endif


// Muscarinic-activated K+ outward current
//
#if CHANNEL_MK == MK_ADAMS_BROWN_CONSTANTI_1982
#define AMC 0.0033
#define AMV 35.0
#define AMD 0.1
#define BMC 0.0033
#define BMV 35.0
#define BMD -0.1
//#define T_ADJ 2.9529 // 2.3^((34-21)/10)
#elif CHANNEL_MK == MK_FUJITA_2012                                                
// Gate: m^4                                                            
//Reference from Fujita et al, Kv7 M-type potassium current
//modification of GPe neuron model proposed by Gunay et al (GPe neuron in basal ganglia)                 
//                                                      
//m activation, h inactivation
// dm/dt = (minf( V ) - V)/tau_m(V) 
// minf  = 1 / (1 + exp( (V - VHALF_M) / k_M))                               
// tau_m = TAU0_M + (TAU1_M - TAU0_M) / ( exp( (PHI_M - V)/SIG0_M) + exp( (PHI_M - V)/SIG1_M) )
//
// NOTE: vtrap(x,y) = x/(exp(x/y)-1)                                                
#define VHALF_M -61

#define k_M 19.5

#define TAU0_M 6.7

#define TAU1_M 100

#define PHI_M -61

#define SIG0_M 35

#define SIG1_M -25



#else
   NOT IMPLEMENTED YET
#endif

void ChannelMK::update(RNG& rng)
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
 
  for (unsigned i = 0; i<branchData->size; ++i) {
    dyn_var_t v = (*V)[i];
#if CHANNEL_MK == MK_ADAMS_BROWN_CONSTANTI_1982
    dyn_var_t am = AMC*exp(AMD*(v + AMV));
    dyn_var_t bm = BMC*exp(BMD*(v + BMV));
    //dyn_var_t pm = 0.5*dt*(am + bm)*T_ADJ; 
    dyn_var_t pm = 0.5*dt*(am + bm)*getSharedMembers().Tadj; 
    //m[i] = (dt*am*T_ADJ + m[i]*(1.0 - pm))/(1.0 + pm);
    m[i] = (dt*am*getSharedMembers().Tadj + m[i]*(1.0 - pm))/(1.0 + pm);
    g[i] = gbar[i]*m[i];
#elif CHANNEL_MK == MK_FUJITA_2012
    {
dyn_var_t taum = TAU0_M + (TAU1_M - TAU0_M) / ( exp( (PHI_M - v)/SIG0_M) + exp( (PHI_M - v)/SIG1_M) );
   dyn_var_t qm = dt  / (taum * 2);
   dyn_var_t m_inf = 1.0 / (1 + exp(( VHALF_M - v) / k_M));
    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    g[i] = gbar[i] * m[i] * m[i] * m[i] * m[i];
    }

#endif

    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);  // at time (t+dt/2)
#if defined(WRITE_GATES)                                                  
    if (is_write)
    {           
      (*outFile) << std::fixed << fieldDelimiter << m[i];                 
    }                                                                     
#endif                                                                    
 
  }
}

void ChannelMK::initialize(RNG& rng)
{
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert(V->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (m.size()!=size) m.increaseSizeTo(size);
  if (Iion.size() != size) Iion.increaseSizeTo(size);
  // initialize
  dyn_var_t gbar_default = gbar[0];
	if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
	{
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels Param"
			<< std::endl;
		assert(0);
	}
  for (unsigned i = 0; i < size; ++i)
  {
    if (gbar_dists.size() > 0) {
      unsigned int j;
      assert(gbar_values.size() == gbar_dists.size());
      for (j=0; j<gbar_dists.size(); ++j) {
        if ((*dimensions)[i]->dist2soma < gbar_dists[j]) break;
      }
      if (j < gbar_values.size()) 
        gbar[i] = gbar_values[j];
      else
        gbar[i] = gbar_default;
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

#if defined(WRITE_GATES)                                      
  if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
      _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
  {
    std::ostringstream os;                                    
    std::string fileName = "gates_MK.txt";                       
    os << fileName << getSimulation().getRank();              
    outFile = new std::ofstream(os.str().c_str());            
    outFile->precision(decimal_places);                       
    (*outFile) << "#Time" << fieldDelimiter << "gates: m [, m]*"; 
    _prevTime = 0.0;                                          
    float currentTime = 0.0;  // should also be (dt/2)                                 
    (*outFile) << std::endl;                                  
    (*outFile) <<  currentTime;                               
  }
#endif
 
  for (unsigned i=0; i<size; ++i) {
    dyn_var_t v = (*V)[i];
#if CHANNEL_MK == MK_ADAMS_BROWN_CONSTANTI_1982
    dyn_var_t am = AMC*exp(AMD*(v + AMV));
    dyn_var_t bm = BMC*exp(BMD*(v + BMV));
    m[i] = am/(am + bm);
    g[i] = gbar[i]*m[i];
#elif CHANNEL_MK == MK_FUJITA_2012 
   m[i] = 1.0 / (1 + exp((VHALF_M-v) / k_M));
   g[i] = gbar[i] * m[i] * m[i] * m[i] * m[i];
#endif

    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);  // at time t0+dt/2
   #if defined(WRITE_GATES)                                      
    if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
        _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
    {
      (*outFile) << std::fixed << fieldDelimiter << m[i];       
    }
#endif                                                        
 
  }
}

ChannelMK::~ChannelMK()
{
}

