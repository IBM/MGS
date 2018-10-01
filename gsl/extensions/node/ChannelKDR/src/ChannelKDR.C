/* =================================================================
 Licensed Materials - Property of IBM
 "Restricted Materials of IBM"

 BMC-YKT-07-18-2017

 (C) Copyright IBM Corp. 2005-2017  All rights reserved

 US Government Users Restricted Rights -
 Use, duplication or disclosure restricted by
 GSA ADP Schedule Contract with IBM Corp.

 =================================================================

 (C) Copyright 2018 New Jersey Institute of Technology.

 =================================================================
*/

#include "Lens.h"
#include "ChannelKDR.h"
#include "CG_ChannelKDR.h"
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



// NOTE: vtrap(x,y) = x/(exp(x/y)-1)
// a_m  = AMC*(V - AMV)/( exp( (V - AMV)/AMD ) - 1.0 )
// b_m  = BMC * exp( (V - BMV)/BMD )
// a_h  = AHC * exp( (V - AHV)/AHD )
// b_h  = BHC / (exp( (V - BHV)/BHD ) + 1.0)

#if CHANNEL_KDR == KDR_HODGKIN_HUXLEY_1952
// data measured from squid giant axon
// Formula:
//    I = gbar * n^4 * (V-Erev)
#define ANC 0.01
#define ANV -55.0
#define AND 10.0
#define BNC 0.125
#define BNV -65.0
#define BND 80.0
#elif CHANNEL_KDR == KDR_SCHWEIGHOFER_1999
// Formula:
//    I = gbar * n^4 * (V-Erev)
// adapted from Rush-Rinzel (1994)
// Rush-Rinzel (1994) thalamic neuron
// adopted from HH-1952 data with
//  the kinetics has been adjusted to 35-degree C using Q10-factor=3

//#define ANC 0.01  
#define ANC 0.2  // This is 0.01/.05 to account for the .05 in \tau_n
#define ANV -41.0
#define AND 10.0
//#define BNC 0.125
#define BNC 2.5  // This is 0.125/.05 to account for the .05 in \tau_n
#define BNV -51.0
#define BND 80.0

#elif CHANNEL_KDR == KDR_TRAUB_1994 || \
      CHANNEL_KDR == KDR_TRAUB_1995
// Formula:
//    I = gbar * n^2 * (V-Erev)
#define Eleak -65.0 //mV
#define ANC 0.016
#define ANV (35.1+Eleak)
#define AND 5.0
#define BNC 0.25
#define BNV (20+Eleak)
#define BND 40.0

#elif CHANNEL_KDR == KDR_WANG_BUSZAKI_1996
// Equations from paper Wang_Buzsaki_1996 
// Formula:
//    I = gbar * n^4 * (V-Erev)
// NOTE: 
//   1. spike threshold ~ -55 mV
//   2. two salient properties of hippocampal neurons + neocortical fast-spiking interneurons --> a brief AHP (about -15 mV measured from the spike threshold) --> Vm repolarized back to about -70mV rather than the E_K = - 90 mV
//    --> KDR needs to be small maximal conductance, and fast gating process so that KDR deactivate quickly during repolarization
#define Vshift 0.0
#define ANC -0.01                                                                  
#define ANV (-34.0+Vshift) 
#define AND -10.0                                                                  
#define BNC 0.125                                                                  
#define BNV (-44.0+Vshift)
#define BND -80.0 

#elif CHANNEL_KDR == KDR_MAHON_2000                                                
// Formula:
//    I = gbar * n^4 * (V-Erev)
// Equations from paper Wang_Buzsaki_1996, half activation voltages shifted by 7mV 
//   designed for Hippocampal interneuron
#define Vshift 7.0 
#define ANC -0.01                                                                  
#define ANV (-34.0+Vshift) 
#define AND -10.0                                                                  
#define BNC 0.125                                                                  
#define BNV (-44.0+Vshift)
#define BND -80.0 
                                                                                   
#elif CHANNEL_KDR == KDR_MIGLIORE_1999
// Migliore, Magee, Hoffman, Johnston (1999) - hippocampal pyramidal neuron
// based on experimental finding in CA1 pyramidal neuron (Hoffman et al., 1997)
// Formula - non-inactivating current
//    I = gbar * n * (Vm - Erev)
#define ANC  1.0
#define ANV  13.0 // mV
#define AND  -9.09  // mV
#define BNC  1.0
#define BNV  13.0  // mV
#define BND  -12.5  // mV
#elif CHANNEL_KDR == KDR_ERISIR_1999
// Designed to model fast-spiking neocortical interneuron 
// Kv1.3 model derived from 'n' type current measured in human T-lymphocyte (Cahalan et al. 1985)
//Formula
//   I = gbar * n^4 
// NOTE: 0.616/0.014 = 44
#define ANC  -0.014
#define ANV -44.0 // mV
#define AND -2.3  // mV
#define BNC 0.0043  
#define BNV -44.0 // mV 
#define BND 34.0  // mV

#elif CHANNEL_KDR == KDR_TUAN_JAMES_2017
// Adopted from Erisir (1999)
// Kv1.3 model derived from 'n' type current measured in human T-lymphocyte (Cahalan et al. 1985)
// Adjusted Vh half-activation
//Formula
//   I = gbar * n^4 
// NOTE: 0.616/0.014 = 44
#define Vh_shift 0.0 //15.0 // mV
//#define scale_tau_n 0.3 //0.4
#define ANC  -0.014
#define ANV (-44.0 + Vh_shift) // mV
#define AND -2.3  // mV
#define BNC 0.0043  
#define BNV (-44.0 + Vh_shift) // mV 
#define BND 34.0  // mV


#elif CHANNEL_KDR == KDR_FUJITA_2012                                                
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


#endif

#ifndef scale_tau_n
#define scale_tau_n 1.0
#endif

// GOAL: update gates using v(t+dt/2) and gate(t-dt/2)
//   --> output gate(t+dt/2)
//   of second-order accuracy at time (t+dt/2) using trapezoidal rule
void ChannelKDR::update(RNG& rng)
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
#if CHANNEL_KDR == KDR_HODGKIN_HUXLEY_1952  
    dyn_var_t an = ANC*vtrap(-(v - ANV), AND);
    dyn_var_t bn = BNC*exp(-(v - BNV)/BND);
    // see Rempe-Chomp (2006)
    dyn_var_t pn = 0.5 * dt * (an + bn) * getSharedMembers().Tadj;
    n[i] = (dt*an*getSharedMembers().Tadj + n[i]*(1.0 - pn))/(1.0 + pn);

    g[i] = gbar[i] * pow(n[i], 4);
    //KDR = gKDR * n^4 * (Vm - Erev)
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);

#elif    CHANNEL_KDR == KDR_SCHWEIGHOFER_1999
    dyn_var_t an = ANC * vtrap(-(v - ANV), AND);
    dyn_var_t bn = BNC * (exp(-(v-BNV)/BND));
    dyn_var_t n_inf = an / (an + bn);
    dyn_var_t qn = 0.5 * dt * (an + bn) * getSharedMembers().Tadj;
    n[i] = (2 * n_inf * qn - n[i] * (qn - 1)) / (qn + 1);
    //KDR = gKDR * n^4 * (Vm - Erev)
    g[i] = gbar[i] * pow(n[i], 4);
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);

#elif CHANNEL_KDR == KDR_TRAUB_1994 || \
      CHANNEL_KDR == KDR_TRAUB_1995
    dyn_var_t an = ANC*vtrap((ANV - v), AND);
    dyn_var_t bn = BNC*exp((BNV - v)/BND);
    // see Rempe-Chomp (2006)
    dyn_var_t pn = 0.5*dt*(an + bn);
    n[i] = (dt*an + n[i]*(1.0 - pn))/(1.0 + pn);
    g[i] = gbar[i]*n[i]*n[i];

    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);

#elif CHANNEL_KDR == KDR_MAHON_2000 || \
      CHANNEL_KDR == KDR_WANG_BUSZAKI_1996                                       
    dyn_var_t an = ANC*vtrap((v - ANV), AND);                              
    dyn_var_t bn = BNC*exp((v - BNV)/BND);                                 
                                                                           
    //dyn_var_t n_inf = an / (an + bn);                                    
                                                                           
    dyn_var_t pn = 0.5 * dt * (an + bn) * getSharedMembers().Tadj;         
    //dyn_var_t qn = 0.5 * dt * (an + bn) * getSharedMembers().Tadj;       
    n[i] = (dt*an*getSharedMembers().Tadj + n[i]*(1.0 - pn)) / (1.0 + pn);   
    //n[i] = (2 * n_inf * qn - n[i] * (qn - 1)) / (qn + 1);                
    //n[i] = phi * (an*(1-n[i])-bn*n[i]);                                  
                                                                           
    g[i] = gbar[i] * pow(n[i], 4);
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]); // at time (t+dt/2) 

#elif CHANNEL_KDR == KDR_MIGLIORE_1999 
    dyn_var_t an = ANC * exp( (v - ANV) / AND );
    dyn_var_t bn = BNC * exp( (v - BNV) / BND );
    dyn_var_t tau_n = std::max(2.0, 50.0 * bn / (bn + an));
    dyn_var_t n_inf = 1.0 / (1 + an);
    dyn_var_t qn = dt * getSharedMembers().Tadj / (tau_n * 2);         
    n[i] = (2 * n_inf * qn - n[i] * (qn - 1)) / (qn + 1);                
    g[i] = gbar[i] * n[i]; 
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
#elif CHANNEL_KDR == KDR_ERISIR_1999 || \
      CHANNEL_KDR == KDR_TUAN_JAMES_2017
    dyn_var_t an = ANC * vtrap( v - ANV, AND);
    dyn_var_t bn = BNC / (exp((v - BNV) / BND));
    dyn_var_t tau_n = scale_tau_n * 1.0 / (an + bn);

//#if  CHANNEL_KDR == KDR_TUAN_JAMES_2017
//#define scale_tau_n_left_side 0.02
//    if (v <= - 60.0)
//      tau_n = tau_n * scale_tau_n_left_side;
//#endif

    dyn_var_t n_inf = an / (an + bn);
    // m' = (m_inf - m ) / tau_m;
    dyn_var_t qn = dt * getSharedMembers().Tadj / (tau_n * 2);
    n[i] = (2 * n_inf * qn - n[i] * (qn - 1)) / (qn + 1);                
    g[i] = gbar[i] * pow(n[i], 4); 
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
 #elif CHANNEL_KDR == KDR_FUJITA_2012
    {
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
    }
#else
    NOT IMPLEMENTED YET;
#endif
#if defined(WRITE_GATES)                                                  
    if (is_write)
    {           
      (*outFile) << std::fixed << fieldDelimiter << n[i];                 
    #if CHANNEL_KDR == KDR_FUJITA_2012
      (*outFile) << std::fixed << fieldDelimiter << h[i];
    #endif
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
void ChannelKDR::initialize(RNG& rng)
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
	#if CHANNEL_KDR == KDR_FUJITA_2012    
    (*outFile) << "#Time" << fieldDelimiter << "gates: m, h [, m,h]*"; 
	#else
    (*outFile) << "#Time" << fieldDelimiter << "gates: m [, m]*"; 
	#endif
    _prevTime = 0.0;                                          
    float currentTime = 0.0;  // should also be (dt/2)                                 
    (*outFile) << std::endl;                                  
    (*outFile) <<  currentTime;                               
  }
#endif
  for (unsigned i=0; i<size; ++i) 
  {
    dyn_var_t v=(*V)[i];
#if CHANNEL_KDR == KDR_HODGKIN_HUXLEY_1952 
    dyn_var_t an = ANC*vtrap(-(v - ANV), AND);
    dyn_var_t bn = BNC*exp(-(v - BNV)/BND);
    n[i] = an/(an + bn); // steady-state value
    g[i]=gbar[i] * pow(n[i], 4);
#elif CHANNEL_KDR == KDR_SCHWEIGHOFER_1999
    dyn_var_t an = ANC*vtrap(-(v - ANV), AND);
    dyn_var_t bn = BNC*exp(-(v - BNV)/BND);
    n[i] = an/(an + bn); // steady-state value
    g[i]=gbar[i] * pow(n[i], 4);
#elif CHANNEL_KDR == KDR_TRAUB_1994  || \
      CHANNEL_KDR == KDR_TRAUB_1995 
    dyn_var_t an = ANC*vtrap((ANV-v), AND);
    dyn_var_t bn = BNC*exp((BNV - v)/BND);
    n[i] = an/(an + bn); // steady-state value
    g[i] = gbar[i]*n[i]*n[i];
#elif CHANNEL_KDR == KDR_MAHON_2000 || \
      CHANNEL_KDR == KDR_WANG_BUSZAKI_1996                   
//v is at time (t0)
// so m and h is also at time t0
// however, as they are at steady-state, the value at time (t0+dt/2)
// does not change
    dyn_var_t an = ANC*vtrap((v - ANV), AND);          
    dyn_var_t bn = BNC*exp((v - BNV)/BND);             
    n[i] = an/(an + bn); // steady-state value         
    g[i]=gbar[i] * pow(n[i], 4);

#elif CHANNEL_KDR == KDR_MIGLIORE_1999
    dyn_var_t an = ANC * exp( (v - ANV) / AND );
    n[i] = 1.0 / (1 + an);
    g[i] = gbar[i] * n[i]; 
#elif CHANNEL_KDR  == KDR_ERISIR_1999 || \
      CHANNEL_KDR == KDR_TUAN_JAMES_2017
    dyn_var_t an = ANC * vtrap( v - ANV, AND);
    dyn_var_t bn = BNC / (exp((v - BNV) / BND));
    n[i] = an / (an + bn);
    g[i] = gbar[i] * pow(n[i], 4);
#elif CHANNEL_KDR == KDR_FUJITA_2012
   n[i] = 1.0 / (1 + exp((VHALF_M-v) / k_M));
   h[i]  = H_MIN + (1-H_MIN) / (1 + exp( (VHALF_H-v) / k_H));                               
   g[i] = gbar[i] * n[i] * n[i] * n[i] * n[i] * h[i];
#endif
#if defined(WRITE_GATES)                                      
    if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
        _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
    {
      (*outFile) << std::fixed << fieldDelimiter << n[i];       
      #if CHANNEL_KDR == KDR_FUJITA_2012
      (*outFile) << std::fixed << fieldDelimiter << h[i];  
      #endif
   } 
#endif                                                        
 
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
  }
}

ChannelKDR::~ChannelKDR()
{
#if defined(WRITE_GATES)
   if (outFile) outFile->close();
#endif
}

