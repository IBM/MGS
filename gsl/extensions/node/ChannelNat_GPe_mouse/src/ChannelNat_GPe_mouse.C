// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#include "Lens.h"
#include "ChannelNat_GPe_mouse.h"
#include "CG_ChannelNat_GPe_mouse.h"
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
//Reference from Fujita et al
//modification of GPe neuron model proposed by Gunay et al (GPe neuron in basal ganglia)                 
//                                                      
//m activation, h inactivation, s slow inactivation
// dm/dt = (minf( V ) - V)/tau_m(V) 
// dh/dt = (hinf( V ) - V)/tau_h(V) 
// ds/dt = (sinf( V ) - V)/tau_s(V) 
//  dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M - Vhalf_m_shift) / k_M));
// minf  = 1 / (1 + exp( (V - VHALF_M) / k_M))                               
// hinf  = 1 / (1 + exp( (V - VHALF_H) / k_H))                               
// sinf  = S_MIN + (1-S_MIN) / (1 + exp( (V - VHALF_S) / k_S))                               
// tau_m = TAU0_M  
// tau_h = TAU0_H + (TAU1_H - TAU0_H) / ( exp( (PHI_H - V)/SIG0_H) + exp( (PHI_H - V)/SIG1_H) )
// tau_s = TAU0_S + (TAU1_S - TAU0_S) / ( exp( (PHI_S - V)/SIG0_S) + exp( (PHI_S - V)/SIG1_S) )
// NOTE: vtrap(x,y) = x/(exp(x/y)-1)                                                
//#define Vshift 7  // [mV]                                                           
#define VHALF_M -39

#define k_M 5.0

#define VHALF_H -48

#define k_H -2.8

#define VHALF_S -40

#define k_S -5.4

#define S_MIN 0.15

#define TAU0_M .028

#define TAU0_H 0.25

#define TAU1_H 4.0

#define PHI_H -43

#define SIG0_H 10

#define SIG1_H -5.0

#define TAU0_S 10

#define TAU1_S 1000

#define PHI_S -40

#define SIG0_S 18.3

#define SIG1_S -10


#ifndef scale_tau_m
#define scale_tau_m 1.0
#endif
#ifndef scale_tau_h
#define scale_tau_h 1.0 
#endif


void ChannelNat_GPe_mouse::update(RNG& rng) 
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
  dyn_var_t taum = TAU0_M;
   dyn_var_t qm = dt  / (taum * 2);
   dyn_var_t tauh = TAU0_H + (TAU1_H - TAU0_H) / ( exp( (PHI_H - v)/SIG0_H) + exp( (PHI_H - v)/SIG1_H) );
   dyn_var_t qh = dt  / (tauh * 2);
   dyn_var_t taus = TAU0_S + (TAU1_S - TAU0_S) / ( exp( (PHI_S - v)/SIG0_S) + exp( (PHI_S - v)/SIG1_S) );
   dyn_var_t qs = dt  / (taus * 2);
   dyn_var_t m_inf = 1.0 / (1 + exp(( VHALF_M - v) / k_M));
   dyn_var_t h_inf = 1.0 / (1 + exp(( VHALF_H - v) / k_H));
   dyn_var_t s_inf  = S_MIN + (1-S_MIN) / (1 + exp( ( VHALF_S - v) / k_S));                               
    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
    s[i] = (2 * s_inf * qs - s[i] * (qs - 1)) / (qs + 1);
     {//keep range [0..1]
    // trick to keep m in [0, 1]
    if (m[i] < 0.0) { m[i] = 0.0; }
    else if (m[i] > 1.0) { m[i] = 1.0; }
    // trick to keep h in [0, 1]
    if (h[i] < 0.0) { h[i] = 0.0; }
    else if (h[i] > 1.0) { h[i] = 1.0; }
    if (s[i] < 0.0) { s[i] = 0.0; }
    else if (s[i] > 1.0) { s[i] = 1.0; }
    }
  
    g[i] = gbar[i] * m[i] * m[i] * m[i] * h[i] * s[i];   
#ifdef WAIT_FOR_REST
    float currentTime = getSimulation().getIteration() * (*getSharedMembers().deltaT);
    if (currentTime < NOGATING_TIME)
      g[i]= 0.0;
#endif
#if defined(WRITE_GATES)                                                  
    if (is_write)
    {           
      (*outFile) << std::fixed << fieldDelimiter << m[i];                 
      (*outFile) << std::fixed << fieldDelimiter << h[i];
      (*outFile) << std::fixed << fieldDelimiter << s[i];
   }
#endif                                                                    
 
}
}
void ChannelNat_GPe_mouse::initialize(RNG& rng) 
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
  if (s.size() != size) s.increaseSizeTo(size);
  //if (Vhalf_m_shift.size() !=size) Vhalf_m_shift.increaseSizeTo(size);
  //if (Vhalf_h_shift.size() !=size) Vhalf_h_shift.increaseSizeTo(size);
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
  // initialize
  SegmentDescriptor segmentDescriptor;
  float gbar_default = gbar[0];
  //float Vhalf_m_default = Vhalf_m_shift[0];
  //float Vhalf_h_default = Vhalf_h_shift[0];
  if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels Nat Param"
      << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
    //Vhalf_m_shift[i] = 0.0; //[mV]
    //Vhalf_h_shift[i] = 0.0; //[mV]
    //Vhalf_m_shift[i] = Vhalf_m_shift_default; //[mV]
    //Vhalf_h_shift[i] = Vhalf_h_shift_default; //[mV]
    //#if CHANNEL_NAT == NAT_COLBERT_PAN_2002
    //		//Vhalf_m init
    //		//NOTE: Shift to the left V1/2 for Nat in AIS region
    //#define DIST_START_AIS   30.0 //[um]
    //		if ((segmentDescriptor.getBranchType(branchData->key) == Branch::_AIS)
    //		// 	or 
    //		//	(		(segmentDescriptor.getBranchType(branchData->key) == Branch::_AXON)  and
    //	  //		 	(*dimensions)[i]->dist2soma >= DIST_START_AIS)
    //			)
    //		{
    //			//gbar[i] = gbar[i] * 1.50; // increase 3x
    //			//Vhalf_m_shift[i] = -15.0 ; //[mV]
    //			//Vhalf_h_shift[i] = -3.0 ; //[mV]
    //		}
    //#endif

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
    (*outFile) << "#Time" << fieldDelimiter << "gates: m, h, s [, m,h,s]*"; 
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
   s[i]  = S_MIN + (1-S_MIN) / (1 + exp( (VHALF_S-v) / k_S));                               
    g[i] = gbar[i] * m[i] * m[i] * m[i] * h[i] * s[i];
    Iion[i] = g[i] * (v - getSharedMembers().E_Na[0]); //using 'v' at time 't'; but gate(t0+dt/2)
#if defined(WRITE_GATES)        
    if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
        _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
    {
      (*outFile) << std::fixed << fieldDelimiter << m[i];       
      (*outFile) << std::fixed << fieldDelimiter << h[i];  
   (*outFile) << std::fixed << fieldDelimiter << s[i];  
    }
#endif                                                        
  }
}

ChannelNat_GPe_mouse::~ChannelNat_GPe_mouse() 
{
#if defined(WRITE_GATES)            
  if (outFile) outFile->close();    
#endif                              


}

