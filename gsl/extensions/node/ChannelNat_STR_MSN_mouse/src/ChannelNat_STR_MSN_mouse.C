// =================================================================
//
// (C) Copyright 2018 New Jersey Institute of Technology.
//
// =================================================================


#include "Lens.h"
#include "ChannelNat_STR_MSN_mouse.h"
#include "CG_ChannelNat_STR_MSN_mouse.h"
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

// Gate: m_inf^3 * h                                                                
//Reference from Wang Buzsaki (FSI neuron in neocortex/hippocampus)                 
//            but voltages shifted by 7mv,                                          
//m is substitute by its steady state value as activation variable m is assumed fast
// m_infty = a_m/(a_m + b_m)
// dh/dt = Phi * (a_h * (1-h) - b_h * h)
// a_m  = AMC*(V - AMV)/( exp( (V - AMV)/AMD  ) - 1.0  )                              
// b_m  = BMC * exp( (V - BMV)/BMD  )                                                
// a_h  = AHC * exp( (V - AHV)/AHD  )                                                
// b_h  = BHC / (exp( (V - BHV)/BHD  ) + 1.0)                                        
// NOTE: gNa = 1.20 nS/um^2 (equivalent to 120 mS/cm^2)                             
//   can be used with Q10 = 3                                                       
// NOTE: vtrap(x,y) = x/(exp(x/y)-1)                                                
#define Vshift 7  // [mV]                                                           
#define AMC -0.1                                                                    
#define AMV (-35.0+Vshift)                                                          
#define AMD -10                                                                     
#define BMC 4.0                                                                     
#define BMV (-60.0+Vshift)                                                          
#define BMD -18.0                                                                   
#define AHC 0.07                                                                    
#define AHV (-58.0+Vshift)                                                          
#define AHD -20.0                                                                   
#define BHC 1.0                                                                     
#define BHV (-28.0+Vshift)                                                          
#define BHD -10.0                                                                   

// GOAL: update gates using v(t+dt/2) and gate(t-dt/2)
//   --> output gate(t+dt/2+dt)
//   of second-order accuracy at time (t+dt/2+dt) using trapezoidal rule
void ChannelNat_STR_MSN_mouse::update(RNG& rng) 
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
    Iion[i] = g[i] * (v - getSharedMembers().E_Na[0]); // at time (t+dt/2)
    // NOTE: Some models use alpha_m and beta_m to estimate m                   
    dyn_var_t am = AMC * vtrap((v - AMV - Vhalf_m_shift), AMD);
    dyn_var_t bm = BMC * exp((v - BMV - Vhalf_m_shift) / BMD);
    m[i] = am / (am + bm);  // steady-state value 
    dyn_var_t ah =  AHC * exp((v - AHV - Vhalf_h_shift) / AHD);  
    dyn_var_t bh =  BHC / (1.0 + exp((v - BHV - Vhalf_h_shift) / BHD));
    dyn_var_t ph = 0.5 * dt * (ah + bh) * getSharedMembers().Tadj;              
    h[i] = (dt * ah * getSharedMembers().Tadj + h[i] * (1.0 - ph)) / (1.0 + ph);
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
void ChannelNat_STR_MSN_mouse::initialize(RNG& rng) 
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

  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];
//v is at time (t0)
// so m and h is also at time t0
// however, as they are at steady-state, the value at time (t0+dt/2)
// does not change
      dyn_var_t am = AMC * vtrap((v - AMV), AMD);       
      dyn_var_t bm = BMC * exp((v - BMV) / BMD);        
      dyn_var_t ah = AHC * exp((v - AHV) / AHD);        
      dyn_var_t bh = BHC / (1.0 + exp((v - BHV) / BHD));
      m[i] = am / (am + bm);  // steady-state value     
      h[i] = ah / (ah + bh);                            
      g[i] = gbar[i] * m[i] * m[i] * m[i] * h[i]; // at time (t+dt/2) - 
  }
}

ChannelNat_STR_MSN_mouse::~ChannelNat_STR_MSN_mouse() 
{
}

