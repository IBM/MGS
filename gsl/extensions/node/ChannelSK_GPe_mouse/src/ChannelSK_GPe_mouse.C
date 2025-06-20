// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#include "Mgs.h"
#include "ChannelSK_GPe_mouse.h"
#include "CG_ChannelSK_GPe_mouse.h"
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

// unit conversion 
#define um2mm  1e-3 // from [um] to [mm] concentration

#if defined(WRITE_GATES)                                      
SegmentDescriptor ChannelSK_GPe_mouse::_segmentDescriptor;
#endif



#define Cai_base 0.1 // [um]
//
// this is an implementation of the "sk(ca) potassium current

// This is an implementation of the SK channel in Fujita et al 2012
// modification of GPe neuron model proposed by Gunay et al.
// m activation
// dm/dt = (minf( [Ca]) - V)/tau_m([Ca])

#define KCa_half .35
#define Hill_coef 4.6
#define TAU_1 76
#define TAU_0 4.0
#define Ca_sat 5.0

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
dyn_var_t ChannelSK_GPe_mouse::KV(dyn_var_t k, dyn_var_t d, dyn_var_t v)
{
  //const dyn_var_t zCa = 2;
  dyn_var_t exp1 = k * exp(-zCa * d * zF * v / zR / (*(getSharedMembers().T)));
  return exp1;
}

// NOTE: 
//    v = voltage [mV]
//    cai = [Ca]cyto [mM]
dyn_var_t  ChannelSK_GPe_mouse::fwrate(dyn_var_t v, dyn_var_t cai)
{
	dyn_var_t fwrate;
	const dyn_var_t d1 = 0.84; // [0..1] unitless
	const dyn_var_t k1 = 0.18; //[mM] - K(0)
	fwrate = beta / (1 + KV(k1,d1,v)/cai);
	return fwrate;
}
dyn_var_t ChannelSK_GPe_mouse::bwrate(dyn_var_t v, dyn_var_t cai)
{
	dyn_var_t bwrate;
	const dyn_var_t d2 = 1.0;// [0..1] unitless
	const dyn_var_t k2 = 0.011; //[mM] - K(0)
	bwrate = alpha / (1 + cai/KV(k2,d2,v));
	return bwrate;
}


void ChannelSK_GPe_mouse::update(RNG& rng) 
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

		Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);

#if defined(WRITE_GATES)                                                  
    if (is_write)
    {           
      (*outFile) << std::fixed << fieldDelimiter << fO[i];                 
   }                                                                     
#endif  
  }


}

void ChannelSK_GPe_mouse::initialize(RNG& rng) 
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

 {
    //m[i] = 1.0/(1 + pow(KCa_half/cai,Hill_coef));
    //g[i] = gbar[i]*m[i];
    fO[i] = 1.0/(1 + pow(KCa_half/cai,Hill_coef));
    g[i] = gbar[i]*fO[i];
    }
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
ChannelSK_GPe_mouse::~ChannelSK_GPe_mouse() 
{
}

