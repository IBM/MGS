// =================================================================
//
// (C) Copyright 2018 New Jersey Institute of Technology.
//
// =================================================================


#include "Lens.h"
#include "ChannelNap_GPe_mouse.h"
#include "CG_ChannelNap_GPe_mouse.h"
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



void ChannelNap_GPe_mouse::update(RNG& rng) 
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
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
    //common
    Iion[i] = g[i] * (v - getSharedMembers().E_Na[0]); // at time (t+dt/2)
  } 
}

void ChannelNap_GPe_mouse::initialize(RNG& rng) 
{
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
   m[i] = 1.0 / (1 + exp((VHALF_M-v) / k_M));
   h[i] = H_MIN + (1-H_MIN) / (1 + exp((VHALF_H-v) / k_H));
   s[i]  = 1.0  / (1 + exp( (VHALF_S-v) / k_S));   
   g[i] = gbar[i]*m[i]*m[i]*m[i]*h[i]*s[i];   
    Iion[i] = g[i] * (v - getSharedMembers().E_Na[0]);
  }

}

ChannelNap_GPe_mouse::~ChannelNap_GPe_mouse() 
{
}

