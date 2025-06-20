// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "ChannelKv31_STR_FSI_mouse.h"
#include "CG_ChannelKv31_STR_FSI_mouse.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"
#include "MaxComputeOrder.h"
#include "NumberUtils.h"

#define SMALL 1.0E-6
#define decimal_places 6     
#define fieldDelimiter "\t"  
#include <math.h>
#include <pthread.h>
#include <algorithm>

// This next block doesn't make sense here... fill in if we ever need to WRITE_GATES
static pthread_once_t once_Nat = PTHREAD_ONCE_INIT;
#if defined(WRITE_GATES)                                      
SegmentDescriptor ChannelNat::_segmentDescriptor;
#endif

// Gate: h^2      (keeping consistent notation with paper)                             //Reference from Corbit et al 2016
//modification of FSI neuron model proposed by Golomb et al 2007 (FSI neuron is in striatum)                 
//                                                      
// h inactivation
// dh/dt = (hinf( V ) - V)/tau_h(V) 
// hinf  = 1 / (1 + exp( (V - VHALF_H) / k_H))                               
// tau_h = (TAU0_H + (TAU1_H - TAU0_H) / ( exp( (PHI1_H - V)/SIG1_H) ) *  (TAU0_H + (TAU1_H - TAU0_H) / ( exp( (PHI2_H - V)/SIG2_H))
// // NOTE: vtrap(x,y) = x/(exp(x/y)-1)                                                
//#define Vshift 7  // [mV]                                                           


#define VHALF_H -12.4

#define k_H 6.8

#define TAU0_H 0.087

#define TAU1_H 11.313

#define PHI1_H -14.6

#define SIG1_H -8.6

#define PHI2_H 1.3

#define SIG2_H 18.7






void ChannelKv31_STR_FSI_mouse::update(RNG& rng) 
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
  dyn_var_t tauh = (TAU0_H + (TAU1_H - TAU0_H) / (1.0+exp(  (PHI1_H - v)/SIG1_H) ) )*  (TAU0_H + (TAU1_H - TAU0_H) / (1.0+ exp( (PHI2_H - v)/SIG2_H) ) );
   dyn_var_t qh = dt  / (tauh * 2.0);
   dyn_var_t h_inf  = 1.0 / (1.0 + exp( ( VHALF_H - v) / k_H));                               
    h[i] = (2.0 * h_inf * qh - h[i] * (qh - 1.0)) / (qh + 1.0); 
#if defined(WRITE_GATES)                                                  
    if (is_write)
    {           
      (*outFile) << std::fixed << fieldDelimiter << h[i];
   }
#endif
    // trick to keep m in [0, 1]
    if (h[i] < 0.0) { h[i] = 0.0; }
    else if (h[i] > 1.0) { h[i] = 1.0; }
    g[i] = gbar[i] * h[i]* h[i];
		Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
  } 
}

void ChannelKv31_STR_FSI_mouse::initialize(RNG& rng) 
{
 //pthread_once(&once_KAf, ChannelKAf::initialize_others);
  assert(branchData);
  unsigned size = branchData->size;
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (h.size() != size) h.increaseSizeTo(size);
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
   h[i]  = 1.0  / (1.0 + exp( (VHALF_H-v) / k_H));       
   g[i] = gbar[i] * h[i] * h[i];
      #if defined(WRITE_GATES)                                      
    if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
        _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
    {
      (*outFile) << std::fixed << fieldDelimiter << h[i];  
   } 
     #endif
Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
 }

}

ChannelKv31_STR_FSI_mouse::~ChannelKv31_STR_FSI_mouse() 
{
}

