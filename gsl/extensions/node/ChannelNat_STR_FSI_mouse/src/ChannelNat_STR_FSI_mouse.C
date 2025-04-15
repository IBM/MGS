// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#include "Lens.h"
#include "ChannelNat_STR_FSI_mouse.h"
#include "CG_ChannelNat_STR_FSI_mouse.h"
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

static pthread_once_t once_Nat = PTHREAD_ONCE_INIT;
#if defined(WRITE_GATES)                                      
SegmentDescriptor ChannelNat::_segmentDescriptor;
#endif

// Gate: m^3 * h * s                                                               
//Reference from Corbit et al 2016
//modification of FSI neuron model proposed by Golomb  et al 2007 (FSI  neuron in striatum)                 
//                                                      
//m instantaneous activation, h inactivation
// dh/dt = (hinf( V ) - V)/tau_h(V) 
//  dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M - Vhalf_m_shift) / k_M));
// minf  = 1 / (1 + exp( (V - VHALF_M) / k_M))                               
// hinf  = 1 / (1 + exp( (V - VHALF_H) / k_H))                               
// tau_h = TAU0_H + (TAU1_H - TAU0_H) / ( exp( (PHI_H - V)/SIG0_H) + exp( (PHI_H - V)/SIG1_H) )
// NOTE: vtrap(x,y) = x/(exp(x/y)-1)                                                
//#define Vshift 7  // [mV]                                                           
#define VHALF_M -24

#define k_M 11.5

#define VHALF_H -58.3

#define k_H -6.7

#define TAU0_H 0.5

#define TAU1_H 13.5

#define PHI_H -60

#define SIG0_H -12



void ChannelNat_STR_FSI_mouse::update(RNG& rng) 
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
    Iion[i] = g[i] * (v - getSharedMembers().E_Na[0]); // at time (t+dt/2)
   dyn_var_t tauh = TAU0_H + (TAU1_H - TAU0_H) / (1.0+ exp( (PHI_H - v)/SIG0_H));
   dyn_var_t qh = dt  / (tauh * 2);
   dyn_var_t h_inf = 1.0 / (1 + exp(( VHALF_H - v) / k_H));
    m[i] = 1.0 / (1.0 + exp(( VHALF_M - v) / k_M));
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
     {//keep range [0..1]
    // trick to keep m in [0, 1]
    if (m[i] < 0.0) { m[i] = 0.0; }
    else if (m[i] > 1.0) { m[i] = 1.0; }
    // trick to keep h in [0, 1]
    if (h[i] < 0.0) { h[i] = 0.0; }
    else if (h[i] > 1.0) { h[i] = 1.0; }
    }
  
    g[i] = gbar[i] * m[i] * m[i] * m[i] * h[i];   
#ifdef WAIT_FOR_REST
    float currentTime = getSimulation().getIteration() * (*getSharedMembers().deltaT);
    if (currentTime < NOGATING_TIME)
      g[i]= 0.0;
#endif
#if defined(WRITE_GATES)                                                  
    if (is_write)
    {           
      (*outFile) << std::fixed << fieldDelimiter << h[i];
   }
#endif                                                                    
 
}

}

void ChannelNat_STR_FSI_mouse::initialize(RNG& rng) 
{
// pthread_once(&once_Nat, initialize_others);
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
  SegmentDescriptor segmentDescriptor;
  float gbar_default = gbar[0];
  if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels Nat Param"
      << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
      //gbar init
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
    std::string fileName = "gates_Nat.txt";                       
    os << fileName << getSimulation().getRank() ;              
    outFile = new std::ofstream(os.str().c_str());            
    outFile->precision(decimal_places);    
    (*outFile) << "#Time" << fieldDelimiter << "gates: h [, h]*"; 
    _prevTime = 0.0;                                          
    float currentTime = 0.0;  // should also be (dt/2)                                 
    (*outFile) << std::endl;                                  
    (*outFile) <<  currentTime;                               
  }
#endif
  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];
   m[i] = 1.0 / (1 + exp((VHALF_M-v) / k_M));
   h[i] = 1.0 / (1 + exp((VHALF_H-v) / k_H));
    g[i] = gbar[i] * m[i] * m[i] * m[i] * h[i];
    Iion[i] = g[i] * (v - getSharedMembers().E_Na[0]); //using 'v' at time 't'; but gate(t0+dt/2)
#if defined(WRITE_GATES)        
    if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
        _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
    {
      (*outFile) << std::fixed << fieldDelimiter << h[i];  
    }
#endif                                                        
  }

}

ChannelNat_STR_FSI_mouse::~ChannelNat_STR_FSI_mouse() 
{
#if defined(WRITE_GATES)
	if (outFile) outFile->close();
#endif 
}

