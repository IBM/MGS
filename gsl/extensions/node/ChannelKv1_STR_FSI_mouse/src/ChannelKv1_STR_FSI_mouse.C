// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#include "Mgs.h"
#include "ChannelKv1_STR_FSI_mouse.h"
#include "CG_ChannelKv1_STR_FSI_mouse.h"
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
SegmentDescriptor ChannelKv1::_segmentDescriptor;
#endif


// Gate: m^3 * h                                                            
//Reference from Corbit  et al 2016, Kv1 slow delayed rectifier
//modification of FSI neuron model proposed by Colomb et al (FSI neuron in striatum)                 
//                                                      
//m activation, h inactivation,
// dm/dt = (minf( V ) - V)/tau_m(V) 
// dh/dt = (hinf( V ) - V)/tau_h(V) 
// minf  = 1 / (1 + exp( (V - VHALF_M) / k_M))                               
// hinf  = 1/ (1 + exp( (V - VHALF_H) / k_H))                               
// tau_m = TAU0_M
// tau_h = TAU0_H
//
// NOTE: vtrap(x,y) = x/(exp(x/y)-1)                                                
#define VHALF_M -50

#define k_M 20.0

#define VHALF_H -70.0

#define k_H -6.0


#define TAU0_M 2.0

#define TAU0_H 150.0


void ChannelKv1_STR_FSI_mouse::update(RNG& rng) 
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
 dyn_var_t taum = TAU0_M;
   dyn_var_t qm = dt  / (taum * 2.0);
  dyn_var_t tauh = TAU0_H;
   dyn_var_t qh = dt  / (tauh * 2.0);
   dyn_var_t m_inf = 1.0 / (1.0 + exp(( VHALF_M - v) / k_M));
   dyn_var_t h_inf  = 1.0 / (1.0 + exp( ( VHALF_H - v) / k_H));                               
    m[i] = (2.0 * m_inf * qm - m[i] * (qm - 1.0)) / (qm + 1.0);
    h[i] = (2.0 * h_inf * qh - h[i] * (qh - 1.0)) / (qh + 1.0); 
#if defined(WRITE_GATES)                                                  
    if (is_write)
    {           
      (*outFile) << std::fixed << fieldDelimiter << m[i];                 
      (*outFile) << std::fixed << fieldDelimiter << h[i];
   }
#endif
    // trick to keep m in [0, 1]
    if (m[i] < 0.0) { m[i] = 0.0; }
    else if (m[i] > 1.0) { m[i] = 1.0; }
    if (h[i] < 0.0) { h[i] = 0.0; }
    else if (h[i] > 1.0) { h[i] = 1.0; }
    g[i] = gbar[i] * m[i] * m[i] * m[i] * h[i];
		Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
  } 

}

void ChannelKv1_STR_FSI_mouse::initialize(RNG& rng) 
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
  if (h.size() != size) h.increaseSizeTo(size);
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
  // initialize
  float gbar_default = gbar[0];
	if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
	{
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels Kv1 Param"
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
   m[i] = 1.0 / (1 + exp((VHALF_M-v) / k_M));
   h[i]  = 1.0  / (1 + exp( (VHALF_H-v) / k_H));       
   g[i] = gbar[i] * m[i] * m[i] * m[i]*  h[i];
      #if defined(WRITE_GATES)                                      
    if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
        _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
    {
      (*outFile) << std::fixed << fieldDelimiter << m[i];       
      (*outFile) << std::fixed << fieldDelimiter << h[i];  
   } 
     #endif
Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
 }

}

ChannelKv1_STR_FSI_mouse::~ChannelKv1_STR_FSI_mouse() 
{
}

