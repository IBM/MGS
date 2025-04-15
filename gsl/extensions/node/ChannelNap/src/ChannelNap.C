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
#include "ChannelNap.h"
#include "CG_ChannelNap.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "GlobalNTSConfig.h"
#include "NumberUtils.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

#include "SegmentDescriptor.h"

static pthread_once_t once_Nap = PTHREAD_ONCE_INIT;

//
// This is an implementation of the "TTX-sensitive slowly-activating, and
// persistent-activating
//         Vm-gated Na^+ current, I_Nap".
//
#if CHANNEL_NAP == NAP_MAGISTRETTI_1999
// Magistretti - Alonso (1999)
//     Slow voltage-inactivation of sustained Na+ current in entorhinal cortex
//     Layer 2 principal pyramidal neurons, i.e. stellate cells 
//     --> whole-cell and single-channel
//     study
#define VHALF_M -52.6
#define k_M -4.6
#define VHALF_H -48.8
#define k_H 10.0

#define IMV 52.6
#define IMD -4.6
#define IHV 48.8
#define IHD 10.0
#define AMC -0.182
#define AMV 38.0
#define AMD -6.0
#define BMC 0.124
#define BMV 38.0
#define BMD 6.0
#define AHC 2.88E-6
#define AHV 17.0
#define AHD 4.63
#define BHC -6.94E-6
#define BHV 64.4
#define BHD -2.63
//#define T_ADJ 2.9529 // 2.3^((34-21)/10)

#elif CHANNEL_NAP == NAP_MAHON_2000   
//MAHON 2000 et al I_Nap = g_bar * m; 
#define VHALF_M -47.8                 
#define k_M -3.1                      
#define tau_m 1                       

#elif CHANNEL_NAP == NAP_WOLF_2005
// data
//    Activation : from 
//      1. Magistretti-Alonso(1999) entorhinal cortical Layer-II stellate cell  (Fig. 4)
//      2. Traub (2003) computational model of Layer2/3 pyramidal neuron - tau_m (Table A2)
//    Inactivation: from 
//      1. Magistretti-Alonso (1999)
//     recorded at 22-24C and then mapped to 35C using Q10 = 3
//
// model is used for simulation NAc nucleus accumbens (medium-sized spiny MSN
// cell)
//    at 35.0 Celcius
// minf(Vm) = 1/(1+exp((Vm-Vh)/k))
// hinf(Vm) = 1/(1+exp(Vm-Vh)/k)
#define VHALF_M -52.6
#define k_M -4.6
#define VHALF_H -48.8
#define k_H 10.0
#define LOOKUP_TAUH_LENGTH 15  // size of the below array
const dyn_var_t ChannelNap::_Vmrange_tauh[] = {-100, -90, -80, -70, -60, -50, -40, -30,
                                    -20,  -10, 0,   10,  20,  30,  40};
dyn_var_t ChannelNap::tauhNap[] = {4500, 4750, 5200, 6100, 6300, 5000, 4250, 3500,
                              3000, 2700, 2500, 2100, 2100, 2100, 2100};
std::vector<dyn_var_t> ChannelNap::Vmrange_tauh;
#elif CHANNEL_NAP == NAP_FUJITA_2012
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
#define VHALF_M -57.7

#define k_M 5.7

#define VHALF_H -57

#define k_H -4

#define H_MIN 0.154

#define VHALF_S -10

#define k_S -4.9

#define TAU0_M 0.03

#define TAU1_M 0.146

#define PHI_M -42.6

#define SIG0_M 14.4

#define SIG1_M -14.4

#define TAU0_H 10

#define TAU1_H 17

#define PHI_H -34

#define SIG0_H 26

#define SIG1_H -31.9

#define AALPHA_S -0.00000288

#define BALPHA_S -0.000049

#define KALPHA_S 4.63

#define ABETA_S 0.00000694

#define BBETA_S 0.000447

#define KBETA_S -2.63

#else
NOT IMPLEMENTED YET
#endif



// GOAL: update gates using v(t+dt/2) and gate(t-dt/2)
//   --> output gate(t+dt/2)
//   of second-order accuracy at time (t+dt/2) using trapezoidal rule
void ChannelNap::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_NAP == NAP_WOLF_2005
    {
    // NOTE: Some models use m_inf and tau_m to estimate m
    dyn_var_t tau_m;
    if (v < -40.0)
      tau_m = 0.025 + 0.14 * exp((v + 40) / 10);
    else
      tau_m = 0.02 + 0.145 * exp(-(v + 40) / 10);
    dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m * 2);
    std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_tauh.begin(), Vmrange_tauh.end(), v);
    int index = low - Vmrange_tauh.begin();
    //dyn_var_t qh = dt * getSharedMembers().Tadj / (tauhNap[index] * 2);
    dyn_var_t tauh;
    if (index == 0)
     tauh = tauhNap[0];
    else if (index < LOOKUP_TAUH_LENGTH)
     tauh = linear_interp(Vmrange_tauh[index-1], tauhNap[index-1], 
        Vmrange_tauh[index], tauhNap[index], v);
    else //assume saturation in taum when Vm > max-value
     tauh = tauhNap[index-1];

    dyn_var_t qh = dt * getSharedMembers().Tadj / (tauh * 2);

    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H) / k_H));

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
    }
#elif CHANNEL_NAP == NAP_MAGISTRETTI_1999
    {
    dyn_var_t m_inf = 1/(1+exp((v + IMV)/IMD));
    dyn_var_t am = AMC*vtrap(v + AMV, AMD);
    dyn_var_t bm = BMC*vtrap(v + BMV, BMD);
    //dyn_var_t pm = 0.5*dt*(am + bm)*T_ADJ/6.0;
    dyn_var_t pm = 0.5*dt*(am + bm)*getSharedMembers().Tadj/6.0;
    m[i] = (2*pm*m_inf + m[i]*(1.0 - pm))/(1.0 + pm);
    dyn_var_t h_inf = 1/(1+exp((v + IHV)/IHD));
    dyn_var_t ah = AHC*vtrap(v + AHV, AHD);
    dyn_var_t bh = BHC*vtrap(v + BHV, BHD);
    //dyn_var_t ph = 0.5*dt*(ah + bh)*T_ADJ; 
    dyn_var_t ph = 0.5*dt*(ah + bh)*getSharedMembers().Tadj; 
    h[i] = (2*ph*h_inf + h[i]*(1.0 - ph))/(1.0 + ph);
    }

#elif CHANNEL_NAP == NAP_MAHON_2000                            
    {                                                          
    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M) / k_M));    
    dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m * 2); 
                                                               
    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);      
    }

 #elif CHANNEL_NAP == NAP_FUJITA_2012
    {

   dyn_var_t taum = TAU0_M + (TAU1_M - TAU0_M) / ( exp( (PHI_M - v)/SIG0_M) + exp( (PHI_M - v)/SIG1_M) );
   dyn_var_t qm = dt  / (taum * 2);
   dyn_var_t tauh = TAU0_H + (TAU1_H - TAU0_H) / ( exp( (PHI_H - v)/SIG0_H) + exp( (PHI_H - v)/SIG1_H) );
   dyn_var_t qh = dt  / (tauh * 2);
   dyn_var_t ah = (AALPHA_S * v + BALPHA_S) / (1 - exp( (v+BALPHA_S/AALPHA_S)/KALPHA_S));
   dyn_var_t bh = (ABETA_S * v + BBETA_S) / (1 - exp( (v+BBETA_S/ABETA_S)/KBETA_S));
   dyn_var_t taus = 1 / (ah + bh);
   dyn_var_t qs = dt  / (taus * 2);
   dyn_var_t m_inf = 1.0 / (1 + exp(( VHALF_M - v) / k_M));
   dyn_var_t h_inf = H_MIN +  (1.0-H_MIN) / (1 + exp(( VHALF_H - v) / k_H));
   dyn_var_t s_inf  =  1.0 / (1 + exp( ( VHALF_S - v) / k_S));                               
    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
    s[i] = (2 * s_inf * qs - s[i] * (qs - 1)) / (qs + 1);
    }

   
#else
    NOT IMPLEMENTED YET
#endif
    {//keep range [0..1]
    // trick to keep m in [0, 1]
    if (m[i] < 0.0) { m[i] = 0.0; }
    else if (m[i] > 1.0) { m[i] = 1.0; }
    // trick to keep h in [0, 1]
    if (h[i] < 0.0) { h[i] = 0.0; }
    else if (h[i] > 1.0) { h[i] = 1.0; }
    }

#if CHANNEL_NAP == NAP_WOLF_2005
    g[i] = gbar[i] * m[i] * h[i];
#elif CHANNEL_NAP == NAP_MAGISTRETTI_1999
    g[i]=gbar[i]*m[i]*m[i]*m[i]*h[i];
#elif CHANNEL_NAP == NAP_MAHON_2000   
    g[i] = gbar[i] * m[i] ;           
#elif CHANNEL_NAP == NAP_FUJITA_2012
    g[i] = gbar[i] * m[i] * m[i] * m[i] * h[i] * s[i];    
#endif
#ifdef WAIT_FOR_REST
    float currentTime = getSimulation().getIteration() * (*getSharedMembers().deltaT);
    if (currentTime < NOGATING_TIME)
      g[i]= 0.0;
#endif
    //common
    Iion[i] = g[i] * (v - getSharedMembers().E_Na[0]); // at time (t+dt/2)
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
void ChannelNap::initialize(RNG& rng)
{
  pthread_once(&once_Nap, ChannelNap::initialize_others);
  assert(branchData);
  unsigned size = branchData->size;
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (m.size() != size) m.increaseSizeTo(size);
  if (h.size() != size) h.increaseSizeTo(size);
#if CHANNEL_NAP == NAP_FUJITA_2012
  if (s.size() != size) s.increaseSizeTo(size);
#endif
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
  // initialize
	SegmentDescriptor segmentDescriptor;
  float gbar_default = gbar[0];
	if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
	{
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels Nap Param"
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
#if CHANNEL_NAP == NAP_WOLF_2005
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));
    h[i] = 1.0 / (1 + exp((v - VHALF_H) / k_H));
    g[i] = gbar[i] * m[i] * h[i];
#elif CHANNEL_NAP == NAP_MAGISTRETTI_1999
    m[i] = 1/(1+exp((v + IMV)/IMD));
    h[i] = 1/(1+exp((v + IHV)/IHD));
    g[i]=gbar[i]*m[i]*m[i]*m[i]*h[i];
#elif CHANNEL_NAP == NAP_MAHON_2000                 
    m[i] = 1.0 / (1 + exp((v - VHALF_M) / k_M));    
    g[i] = gbar[i] * m[i] ;   
#elif CHANNEL_NAT == NAT_FUJITA_2012
   m[i] = 1.0 / (1 + exp((VHALF_M-v) / k_M));
   h[i] = H_MIN + (1-H_MIN) / (1 + exp((VHALF_H-v) / k_H));
   s[i]  = 1.0  / (1 + exp( (VHALF_S-v) / k_S));   
   g[i] = gbar[i]*m[i]*m[i]*m[i]*h[i]*s[i];   
#else
    NOT IMPLEMENTED YET
#endif
    Iion[i] = g[i] * (v - getSharedMembers().E_Na[0]);
  }
}

void ChannelNap::initialize_others()
{
#if CHANNEL_NAP == NAP_WOLF_2005
  std::vector<dyn_var_t> tmp(_Vmrange_tauh, _Vmrange_tauh + LOOKUP_TAUH_LENGTH);
  assert((sizeof(tauhNap) / sizeof(tauhNap[0])) == tmp.size());
	//Vmrange_tauh.resize(tmp.size()-2);
  //for (unsigned long i = 1; i < tmp.size() - 1; i++)
  //  Vmrange_tauh[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
  Vmrange_tauh = tmp;
#endif
}

ChannelNap::~ChannelNap() {}
