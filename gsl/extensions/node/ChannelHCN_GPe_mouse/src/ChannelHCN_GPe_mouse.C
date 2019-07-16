// =================================================================
//
// (C) Copyright 2018 New Jersey Institute of Technology.
//
// =================================================================


#include "Lens.h"
#include "ChannelHCN_GPe_mouse.h"
#include "CG_ChannelHCN_GPe_mouse.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"
#include "NumberUtils.h"
#include "MaxComputeOrder.h"
#include "Branch.h"

#include <math.h>
#include <pthread.h>
#include <algorithm>

#define SMALL 1.0E-6
#define decimal_places 6     
#define fieldDelimiter "\t"  

#if defined(WRITE_GATES)                                      
SegmentDescriptor ChannelHCN::_segmentDescriptor;
#endif

// Gate: m                                                               
//Reference from Fujita et al, 
// Hyperpolarization-activated cyclic nucleotide-modulated cation current
//modification of GPe neuron model proposed by Gunay et al (GPe neuron in basal ganglia)                 
//                                                      
//m activation
// dm/dt = (minf( V ) - V)/tau_m(V) 
//  dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M - Vhalf_m_shift) / k_M));
// minf  = 1 / (1 + exp( (V - VHALF_M) / k_M))                               
// tau_h = TAU0_M + (TAU1_M - TAU0_M) / ( exp( (PHI_M - V)/SIG0_M) + exp( (PHI_M - V)/SIG1_M) )
// NOTE: vtrap(x,y) = x/(exp(x/y)-1)                                                
//#define Vshift 7  // [mV]                                                           
#define VHALF_M -76.4

#define k_M -3.3

#define TAU0_M 0

#define TAU1_M 3625

#define PHI_M -76.4

#define SIG0_M 6.56

#define SIG1_M -7.48

dyn_var_t ChannelHCN_GPe_mouse::conductance(int i)
{
	//conductance density gh is distributed across compartments
	//using the exponential functions: gh = y0 + A * exp(d/lambda)
	//with y0 = -2 pS/um^2
	//     A  = 4.28 pS/um^2
	//     lambda = 323 um
	//     d  = distance from the soma
	float d = (*dimensions)[i]->dist2soma;
	const float y0 = -2.0 ;
	const float A = 4.28 ; 
	const float	lambda = 323.0 ;
	dyn_var_t gh = y0 + A * exp(d/lambda);
	return gh;
}



void ChannelHCN_GPe_mouse::update(RNG& rng) 
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
     {
   dyn_var_t taum = TAU0_M + (TAU1_M - TAU0_M) / ( exp( (PHI_M - v)/SIG0_M) + exp( (PHI_M - v)/SIG1_M) );
   dyn_var_t qm = dt  / (taum * 2);
   dyn_var_t m_inf = 1.0 / (1 + exp(( VHALF_M - v) / k_M));
    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    g[i] = gbar[i] * m[i];
    }

		Iion[i] = g[i] * (v - getSharedMembers().E_HCN[0]);
  #if defined(WRITE_GATES)                                                  
    if (is_write)
    {           
      (*outFile) << std::fixed << fieldDelimiter << m[i];                 
   }                                                                     
#endif                                                                    
  } 
}

void ChannelHCN_GPe_mouse::initialize(RNG& rng) 
{
  unsigned size = branchData->size;
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);

  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (m.size() != size) m.increaseSizeTo(size);
  if (Iion.size()!=size) Iion.increaseSizeTo(size);

  // initialize
  // NOTE: add the scaling_factor for testing channel easier
  //  Should be set to 1.0 by default
  //dyn_var_t scaling_factor = 1.0;
  //float gbar_default = gbar[0] * scaling_factor;
  float gbar_default = gbar[0] ;
  //float gbar_default = gbar[0];
  if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
  {
    std::cerr
        << "ERROR: Use either gbar_dists or gbar_branchorders on Channels HCN Param"
        << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
    if (gbar_dists.size() > 0)
    {
      unsigned int j;
	  //NOTE: 'n' bins are splitted by (n-1) points
	  //gbar_dists = hold such points
	  //gbar_values = hold value in each bin
      if (gbar_values.size() - 1 != gbar_dists.size())
      {
        std::cerr << "gbar_values.size = " << gbar_values.size() 
          << "; gbar_dists.size = " << gbar_dists.size() << std::endl; 
      }
      assert(gbar_values.size() -1 == gbar_dists.size());
      for (j = 0; j < gbar_dists.size(); ++j)
      {
        if ((*dimensions)[i]->dist2soma < gbar_dists[j]) break;
      }
      gbar[i] = gbar_values[j] ;
    }
    else if (gbar_branchorders.size() > 0)
    {
      unsigned int j;
      assert(gbar_values.size() == gbar_branchorders.size());
      SegmentDescriptor segmentDescriptor;
      for (j = 0; j < gbar_branchorders.size(); ++j)
      {
        if (segmentDescriptor.getBranchOrder(branchData->key) ==
            gbar_branchorders[j])
          break;
      }
      if (j == gbar_branchorders.size() and
          gbar_branchorders[j - 1] == GlobalNTS::anybranch_at_end)
      {
        gbar[i] = gbar_values[j - 1] ;
      }
      else if (j < gbar_values.size())
        gbar[i] = gbar_values[j] ;
      else
        gbar[i] = gbar_default;
    }
    else
    {
      gbar[i] = gbar_default;
    }

    gbar[i] = gbar[i]; // * gbar_scale;
  }

#if defined(WRITE_GATES)                                      
  if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
      _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
  {
    std::ostringstream os;                                    
    std::string fileName = "gates_HCN.txt";                       
    os << fileName << getSimulation().getRank() ;              
    outFile = new std::ofstream(os.str().c_str());            
    outFile->precision(decimal_places);    
(*outFile) << "#Time" << fieldDelimiter << "gates: m[, m]*"; 
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
   g[i] = gbar[i] * m[i];
		Iion[i] = g[i] * (v - getSharedMembers().E_HCN[0]);
	#if defined(WRITE_GATES)                                      
    if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
        _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
    {
      (*outFile) << std::fixed << fieldDelimiter << m[i];       
    }
#endif                                                        
 	
}

}
ChannelHCN_GPe_mouse::~ChannelHCN_GPe_mouse() 
{
}

