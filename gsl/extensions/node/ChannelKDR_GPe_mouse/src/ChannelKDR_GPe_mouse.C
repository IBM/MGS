// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#include "Lens.h"
#include "ChannelKDR_GPe_mouse.h"
#include "CG_ChannelKDR_GPe_mouse.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"
#include "MaxComputeOrder.h"
#include "NumberUtils.h"

// NOTE: play a major role in setting interspike interval under constant Iinject
//       and shape late repolarization phase of AP --> prevent doublet
#define SMALL 1.0E-6
#define decimal_places 6
#define fieldDelimiter "\t"
#include <math.h>
#include <pthread.h>
#include <algorithm>

static pthread_once_t once_KDR = PTHREAD_ONCE_INIT;
#if defined(WRITE_GATES)                                      
SegmentDescriptor ChannelKDR::_segmentDescriptor;
#endif

// Gate: m^4 * h                                                            
//Reference from Fujita et al, Kv2 slow delayed rectifier
//modification of GPe neuron model proposed by Gunay et al (GPe neuron in basal ganglia)                 
//                                                      
//m activation, h inactivation
// dm/dt = (minf( V ) - V)/tau_m(V) 
// dh/dt = (hinf( V ) - V)/tau_h(V) 
// minf  = 1 / (1 + exp( (V - VHALF_M) / k_M))                               
// hinf  = H_MIN + (1-H_MIN) / (1 + exp( (V - VHALF_H) / k_H))                               
// tau_m = TAU0_M + (TAU1_M - TAU0_M) / ( exp( (PHI_M - V)/SIG0_M) + exp( (PHI_M - V)/SIG1_M) )
// tau_h = TAU0_H 
//
// NOTE: vtrap(x,y) = x/(exp(x/y)-1)                                                
#define VHALF_M -33.2

#define k_M 9.1

#define VHALF_H -20

#define k_H -10

#define H_MIN 0.2

#define TAU0_M 0.1

#define TAU1_M 3.0

#define PHI_M -33.2

#define SIG0_M 21.7

#define SIG1_M -13.9

#define TAU0_H 3400




void ChannelKDR_GPe_mouse::update(RNG& rng) 
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
 
  for (unsigned i=0; i<branchData->size; ++i) 
  {
    dyn_var_t v=(*V)[i];
 dyn_var_t taum = TAU0_M + (TAU1_M - TAU0_M) / ( exp( (PHI_M - v)/SIG0_M) + exp( (PHI_M - v)/SIG1_M) );
   dyn_var_t qm = dt  / (taum * 2);
   dyn_var_t tauh = TAU0_H;
   dyn_var_t qh = dt  / (tauh * 2);
   dyn_var_t m_inf = 1.0 / (1 + exp(( VHALF_M - v) / k_M));
   dyn_var_t h_inf  = H_MIN + (1-H_MIN) / (1 + exp( ( VHALF_H - v) / k_H));                               
   // in paper, m,h used. Naming convention for m retained in equation, but switch to n 
   // for consistency with regard to data storage
    n[i] = (2 * m_inf * qm - n[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
    g[i] = gbar[i] * n[i] * n[i] * n[i] * n[i] * h[i];
    Iion[i] = g[i] * (v-getSharedMembers().E_K[0]);
#if defined(WRITE_GATES)                                                  
    if (is_write)
    {           
      (*outFile) << std::fixed << fieldDelimiter << n[i];                 
      (*outFile) << std::fixed << fieldDelimiter << h[i];
   }
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

void ChannelKDR_GPe_mouse::initialize(RNG& rng) 
{
  assert(branchData);
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert (V->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (n.size()!=size) n.increaseSizeTo(size);
#if CHANNEL_KDR == KDR_FUJITA_2012
  if (h.size()!=size) h.increaseSizeTo(size);
#endif
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
  // initialize
  SegmentDescriptor segmentDescriptor;

  float gbar_default = gbar[0];
  if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels KDR Param"
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
    std::string fileName = "gates_KDR.txt";                       
    os << fileName << getSimulation().getRank() ;              
    outFile = new std::ofstream(os.str().c_str());            
    outFile->precision(decimal_places);    
    (*outFile) << "#Time" << fieldDelimiter << "gates: m, h [, m,h]*"; 
    _prevTime = 0.0;                                          
    float currentTime = 0.0;  // should also be (dt/2)                                 
    (*outFile) << std::endl;                                  
    (*outFile) <<  currentTime;                               
  }
#endif
  for (unsigned i=0; i<size; ++i) 
  {
    dyn_var_t v=(*V)[i];
  n[i] = 1.0 / (1 + exp((VHALF_M-v) / k_M));
   h[i]  = H_MIN + (1-H_MIN) / (1 + exp( (VHALF_H-v) / k_H));                               
   g[i] = gbar[i] * n[i] * n[i] * n[i] * n[i] * h[i];
#if defined(WRITE_GATES)                                      
    if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
        _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
    {
      (*outFile) << std::fixed << fieldDelimiter << n[i];       
      (*outFile) << std::fixed << fieldDelimiter << h[i];  
   } 
#endif                                                        
 
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
 
}
}
ChannelKDR_GPe_mouse::~ChannelKDR_GPe_mouse() 
{
}

