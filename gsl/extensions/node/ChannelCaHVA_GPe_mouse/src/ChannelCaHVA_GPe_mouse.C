// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#include "Lens.h"
#include "ChannelCaHVA_GPe_mouse.h"
#include "CG_ChannelCaHVA_GPe_mouse.h"
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


//static pthread_once_t once_CaHVA = PTHREAD_ONCE_INIT;
#if defined(WRITE_GATES)                                      
SegmentDescriptor ChannelCaHVA::_segmentDescriptor;
#endif

// Gate: m                                                            
//Reference from Fujita et al, ICaH high-voltage activated calcium current
//modification of GPe neuron model proposed by Gunay et al (GPe neuron in basal ganglia)                 
//                                                      
// m activation
// dm/dt = (minf( V ) - V)/tau_m(V) 
// minf  = 1 / (1 + exp( (V - VHALF_M) / k_M))                               
// tau_m = TAU0_M 
//
// NOTE: vtrap(x,y) = x/(exp(x/y)-1)                                                
#define VHALF_M -20

#define k_M 7
#define Erev_Ca 130
#define TAU0_M 0.2


void ChannelCaHVA_GPe_mouse::initialize(RNG& rng) 
{
 assert(branchData);
  unsigned size = branchData->size;
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (s.size() != size) s.increaseSizeTo(size);
 if (I_Ca.size() != size) I_Ca.increaseSizeTo(size);
  //NOTE I_Ca plays the role for Iion already
  //if (Iion.size()!=size) Iion.increaseSizeTo(size);
  if (E_Ca.size() != size) E_Ca.increaseSizeTo(size);
  // initialize
	SegmentDescriptor segmentDescriptor;
  float gbar_default = gbar[0];
	if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
	{
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels CaHVA Param"
			<< std::endl;
		assert(0);
	}
  for (unsigned i = 0; i < size; ++i)
  {
		//gbar init
		if (gbar_dists.size() > 0) 
    {
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
    std::string fileName = "gates_CaH.txt";                       
    os << fileName << getSimulation().getRank();              
    outFile = new std::ofstream(os.str().c_str());            
    outFile->precision(decimal_places);                       
    (*outFile) << "#Time" << fieldDelimiter << "gates: m [, m]*"; 
    _prevTime = 0.0;                                          
    float currentTime = 0.0;  // should also be (dt/2)                                 
    (*outFile) << std::endl;                                  
    (*outFile) <<  currentTime;                               
  }
#endif
 
  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];
    dyn_var_t cai = (*Ca_IC)[i];
  E_Ca[i] = Erev_Ca;
   s[i] = 1.0 / (1 + exp((VHALF_M-v) / k_M));
   g[i] = gbar[i] * s[i];     
  I_Ca[i] = g[i] * (v-E_Ca[i]);
    //Iion[i] = I_Ca[i];
    //Iion[i] = g[i] * (v-E_Ca[i]);
 #if defined(WRITE_GATES)                                      
    if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
        _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
    {
      (*outFile) << std::fixed << fieldDelimiter << s[i];       
    }
#endif                                                        
 
  }

}

void ChannelCaHVA_GPe_mouse::update(RNG& rng) 
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
    dyn_var_t cai = (*Ca_IC)[i];
  {
	    // s = m
  E_Ca[i] = Erev_Ca;
 dyn_var_t taum = TAU0_M;
   dyn_var_t qm = dt  / (taum * 2);
   dyn_var_t m_inf = 1.0 / (1 + exp(( VHALF_M - v) / k_M));
    s[i] = (2 * m_inf * qm - s[i] * (qm - 1)) / (qm + 1);
    }
    // trick to keep s in [0, 1]
    if (s[i] < 0.0) { s[i] = 0.0; }
    else if (s[i] > 1.0) { s[i] = 1.0; }

 g[i] = gbar[i] * s[i];
    I_Ca[i] = g[i] * (v-E_Ca[i]);

#ifdef WAIT_FOR_REST
		float currentTime = getSimulation().getIteration() * (*getSharedMembers().deltaT);
		if (currentTime < NOGATING_TIME)
			I_Ca[i] = 0.0;
#endif
    //Iion[i] = I_Ca[i]; //g[i] * (v-E_Ca[i]);
#if defined(WRITE_GATES)                                                  
    if (is_write)
    {           
      (*outFile) << std::fixed << fieldDelimiter << s[i];                 
    }                                                                     
#endif                                                                    
 
  }

}

ChannelCaHVA_GPe_mouse::~ChannelCaHVA_GPe_mouse() 
{
}

