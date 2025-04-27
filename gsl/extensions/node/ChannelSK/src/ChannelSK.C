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
#include "ChannelSK.h"
#include "CG_ChannelSK.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"
#include "MaxComputeOrder.h"
#include "NumberUtils.h"

#define SMALL 1.0E-6
#define decimal_places 6     
#define fieldDelimiter "\t"  

// unit conversion 
#ifndef uM2mM
#define uM2mM  1e-3 // from [uM] to [mM] concentration
#endif

#include <math.h>
#include <pthread.h>
#include <algorithm>

#if defined(WRITE_GATES)                                      
SegmentDescriptor ChannelSK::_segmentDescriptor;
#endif



#define Cai_base 0.1 // [uM]
//
// This is an implementation of the "SK(ca) potassium current
//
#if CHANNEL_SK == SK_TRAUB_1994
// SK equation used in many Traub papers:
//   alpha = min(0.2x10^-4 x [Ca]i, 0.01)
//   beta = 0.001
//   chi means [Ca]i
#define alphamax 0.01
#define beta 0.001
//original Traub: chiscaling (0.0002) with Camax(400-den; 40-soma)
//#define chiscaling 0.00002
#define chiscaling 0.0008

#elif CHANNEL_SK == SK2_KOHLER_ADELMAN_1996_RAT
// Kohler -...-Adelman (1996) Science 
//   Small-conductance, Ca2+ activated potassium channels from mammalian brain
//   1. SK permeate to K+ rather than Na+
//   2. record data for hSK1 and rSK2
//   3. current activate instantly and show no inactivation (during 500ms test pulse)
//   SK2 and SK3 is apamin-sensitive
//   SK1 is apamin-insensitive
//   rSK2: gmax = 3.6 [nS]
//   hSK1: gmax = 13.5 [nS]
//  NOTE: Hill_coeff == steepness
#define KCa_half  0.43  // [uM]
#define Hill_coef  4.8 // 4.8+/-1.46  
#define TAU 1.0

#elif CHANNEL_SK == SK1_KOHLER_ADELMAN_1996_HUMAN
#define KCa_half  0.71  // [uM]
#define Hill_coef  3.9  // 3.9+/-0.45 suggest 4 Ca2+ binding sites involved
#define TAU 1.0

#elif CHANNEL_SK == SK_WOLF_2005
// The implementation is indeed from Moczydlowski - Latorre 1993
//  Use Model 3 - Scheme 1 
//    for Ca-dependent only K+ channel from rat skeletal muscle
//    (single channel current)
//
//               C <->C.Ca <=> O.Ca <-> O.Ca2
//   C = C, C.Ca
//   O = O.Ca, O.Ca2
//  which is mapped to Hodgkin-Huxley-type formula using
//  steady-state modeling:
//     (C,C.Ca) <=>[fwrate][bwrate] (O.Ca, O.Ca2)
//     only look at Po at equilibrium
//     Po = fwrate / (fwrate + bwrate)
#define alpha 0.28  // [1/ms]
#define beta 0.480  // [1/ms]

#elif CHANNEL_SK == SK_FUJITA_2012
// This is an implementation of the SK channel in Fujita et al 2012
// modification of GPe neuron model proposed by Gunay et al.
// m activation
// dm/dt = (minf( [Ca]) - V)/tau_m([Ca])

#define KCa_half .35
#define Hill_coef 4.6
#define TAU_1 76
#define TAU_0 4.0
#define Ca_sat 5.0

#else
NOT IMPLEMENTED YET
#endif

#ifndef  alpha
#define alpha 0.00001  // [1/ms]
#endif
#ifndef beta
#define beta 0.000010  // [1/ms]
#endif
// GOAL: find K(Vm) = K(0) * exp(-zCa * delta * F.Vm / (RT))
//   - dissociation constant as a function of voltage
// Unit on return: mM
// k (mM), 
// d=delta=fractional distance of the electric field that is felt by the ions
// Vm (mV)
// F=Faraday
// R=universial constant
// T=temperature
dyn_var_t ChannelSK::KV(dyn_var_t k, dyn_var_t d, dyn_var_t v)
{
  //const dyn_var_t zCa = 2;
  dyn_var_t exp1 = k * exp(-zCa * d * zF * v / zR / (*(getSharedMembers().T)));
  return exp1;
}

// NOTE: 
//    v = voltage [mV]
//    cai = [Ca]cyto [mM]
dyn_var_t  ChannelSK::fwrate(dyn_var_t v, dyn_var_t cai)
{
	dyn_var_t fwrate;
	const dyn_var_t d1 = 0.84; // [0..1] unitless
	const dyn_var_t k1 = 0.18; //[mM] - K(0)
	fwrate = beta / (1 + KV(k1,d1,v)/cai);
	return fwrate;
}
dyn_var_t ChannelSK::bwrate(dyn_var_t v, dyn_var_t cai)
{
	dyn_var_t bwrate;
	const dyn_var_t d2 = 1.0;// [0..1] unitless
	const dyn_var_t k2 = 0.011; //[mM] - K(0)
	bwrate = alpha / (1 + cai/KV(k2,d2,v));
	return bwrate;
}

void ChannelSK::update(RNG& rng)
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
    dyn_var_t v = (*V)[i];      //[mV]
#if ! defined(SIMULATE_CACYTO)
    dyn_var_t cai = Cai_base;
#else
    dyn_var_t cai = (*Cai)[i] ;  //[uM]
#endif

#if CHANNEL_SK == SK_TRAUB_1994
    {
    // a is alpha
    dyn_var_t aq = ((chiscaling*cai)>alphamax)?alphamax:(chiscaling*cai);
    dyn_var_t bq = beta;
    // Rempe & Chopp (2006)
    dyn_var_t pq = 0.5*dt*(aq+bq);
    fO[i] = (dt*aq + fO[i]*(1.0-pq))/(1.0+pq);
    g[i] = gbar[i]*fO[i];
    }
#elif CHANNEL_SK == SK_WOLF_2005
    {//to group code
    // Rate k1-k2: unit 1/(ms)
    dyn_var_t cai_mM = cai * uM2mM; //[mM]

    dyn_var_t a = fwrate(v, cai_mM);
    dyn_var_t sum = a+bwrate(v,cai_mM);
    dyn_var_t tau = 1/(sum);
    dyn_var_t Oinf = a/(sum); 
    dyn_var_t Tscale_tau = 0.5 * dt  * getSharedMembers().Tadj /tau;
    //dO = (Oinf - O)/(tau/Tadj);
    // Rempe-Chopp 2006
    fO[i] = (2 * Oinf * Tscale_tau + fO[i] * (1-Tscale_tau))/(1+Tscale_tau);

#ifdef DEBUG_ASSERT
    //assert(fabs(fO[0] + fC[0] - 1.0) < SMALL);  // conservation
#endif
    // trick to keep fO in [0, 1]
    if (fO[i] < 0.0) { fO[i] = 0.0; }
    else if (fO[i] > 1.0) { fO[i] = 1.0; }
    // trick to keep fC in [0, 1]
    // if (fC[i] < 0.0) {	fC[i] = 0.0;}
    // else if (fC[i] > 1.0) {fC[i] = 1.0;}
    // fC[i] = ...
    // fC[i] = 1.0 - (fO[i]);  //no need
    g[i] = gbar[i] * fO[i] ;
    }
#elif CHANNEL_SK == SK2_KOHLER_ADELMAN_1996_RAT || \
	  CHANNEL_SK == SK1_KOHLER_ADELMAN_1996_HUMAN
  {
	// Rempe-Chopp 2006
	dyn_var_t minf = (1.0)/ (1.0 + pow(KCa_half/cai, Hill_coef));
	dyn_var_t qm = 0.5 * dt * getSharedMembers().Tadj/ TAU ;
	//fO means 'm' (activation gate)
    //m[i] = (2 * minf * qm - m[i] * (qm-1)) / (qm + 1);
    //g[i] = gbar[i]*m[i];
    fO[i] = (2 * minf * qm - fO[i] * (qm-1)) / (qm + 1);
    g[i] = gbar[i]*fO[i];
  }

#elif CHANNEL_SK == SK_FUJITA_2012
   { 	
	   dyn_var_t taum=TAU_0;
		// fO = m
	if (cai<Ca_sat)  {
	 taum = TAU_1 - ( ( TAU_1 - TAU_0 ) * cai ) / Ca_sat; 
	}
//	else {
//	       	dyn_var_t taum = TAU_0;
//	}	
 	   dyn_var_t qm = dt  / (taum * 2);
 	   dyn_var_t m_inf = (1.0)/ (1.0 + pow(KCa_half/cai, Hill_coef));
       	   fO[i] = (2 * m_inf * qm - fO[i] * (qm - 1)) / (qm + 1);
	   g[i] = gbar[i]*fO[i];
    }

#else
    NOT IMPLEMENTED YET
#endif
		Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);

#if defined(WRITE_GATES)                                                  
    if (is_write)
    {           
      (*outFile) << std::fixed << fieldDelimiter << fO[i];                 
   }                                                                     
#endif  
  }
}

void ChannelSK::initialize(RNG& rng)
{
  assert(branchData);
  unsigned size = branchData->size;
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (fO.size() != size) fO.increaseSizeTo(size);
  //if (fC.size() != size) fC.increaseSizeTo(size);
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
  // initialize
	SegmentDescriptor segmentDescriptor;
  float gbar_default = gbar[0];
	if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
	{
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels SK Param"
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
  //assert(fabs(fO[0] + fC[0] - 1.0) < SMALL);  // conservation
  #if defined(WRITE_GATES)                                      
  if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
      _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
  {
    std::ostringstream os;                                    
    std::string fileName = "gates_SK.txt";                       
    os << fileName << getSimulation().getRank() ;              
    outFile = new std::ofstream(os.str().c_str());            
    outFile->precision(decimal_places);    
(*outFile) << "#Time" << fieldDelimiter << "gates: fO [, m]*"; 
    _prevTime = 0.0;                                          
    float currentTime = 0.0;  // should also be (dt/2)                                 
    (*outFile) << std::endl;                                  
    (*outFile) <<  currentTime;                               
  }
#endif

  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];
    //g[i] = gbar[i] * fO[i];
    //fO[i] = fO[0];
#if ! defined(SIMULATE_CACYTO)
    dyn_var_t cai = Cai_base;
#else
    dyn_var_t cai = (*Cai)[i]; // [uM]
#endif

#if CHANNEL_SK == SK_TRAUB_1994
    {
    // a is alpha
    dyn_var_t aq = ((chiscaling*cai)>alphamax)?alphamax:(chiscaling*cai);
    dyn_var_t bq = beta;
    fO[i] = aq/(aq+bq);
    g[i] = gbar[i]*fO[i];
      
    }
#elif CHANNEL_SK == SK2_KOHLER_ADELMAN_1996_RAT || \
    CHANNEL_SK == SK1_KOHLER_ADELMAN_1996_HUMAN || \
    CHANNEL_SK == SK_FUJITA_2012 

    {
    //m[i] = 1.0/(1 + pow(KCa_half/cai,Hill_coef));
    //g[i] = gbar[i]*m[i];
    fO[i] = 1.0/(1 + pow(KCa_half/cai,Hill_coef));
    g[i] = gbar[i]*fO[i];
    }
#elif CHANNEL_SK == SK_WOLF_2005
    {
    dyn_var_t cai_mM = cai  * uM2mM; // [mM]
    dyn_var_t a = fwrate(v, cai_mM);
    dyn_var_t sum = a+bwrate(v,cai_mM);
    //dyn_var_t tau = 1/(sum);
    //dyn_var_t Oinf = a/(sum); 
    //fO[i] = Oinf;
    fO[i] = a/(sum); 
    g[i] = gbar[i]*fO[i];
    }
#endif
		Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);

#if defined(WRITE_GATES)        
    if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
        _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
    {
      (*outFile) << std::fixed << fieldDelimiter << fO[i];       
    }
#endif                                                        
 
  }
}

ChannelSK::~ChannelSK() {}
