// =================================================================
//
// (C) Copyright 2018 New Jersey Institute of Technology.
//
// =================================================================


#include "Lens.h"
#include "ChannelKAs_GPe_mouse.h"
#include "CG_ChannelKAs_GPe_mouse.h"
#include "rndm.h"

#include "GlobalNTSConfig.h"
#include "SegmentDescriptor.h"
#include "Branch.h"
#include <pthread.h>

#define SMALL 1.0E-6
#define decimal_places 6     
#define fieldDelimiter "\t"  

static pthread_once_t once_KAs = PTHREAD_ONCE_INIT;
#if defined(WRITE_GATES)                                      
SegmentDescriptor ChannelKAs::_segmentDescriptor;
#endif

// Gate: m^4 * h                                                            
//Reference from Fujita et al, Kv4s slow inactivating  type A
//modification of GPe neuron model proposed by Gunay et al (GPe neuron in basal ganglia)                 
//                                                      
//m activation, h inactivation
// dm/dt = (minf( V ) - V)/tau_m(V) 
// dh/dt = (hinf( V ) - V)/tau_h(V) 
// minf  = 1 / (1 + exp( (V - VHALF_M) / k_M))                               
// hinf  = 1 / (1 + exp( (V - VHALF_H) / k_H))                               
// tau_m = TAU0_M + (TAU1_M - TAU0_M) / ( exp( (PHI_M - V)/SIG0_M) + exp( (PHI_M - V)/SIG1_M) )
// tau_h = TAU0_H + (TAU1_H - TAU0_H) / ( exp( (PHI_H - V)/SIG0_H) + exp( (PHI_H - V)/SIG1_H) )
//
// NOTE: vtrap(x,y) = x/(exp(x/y)-1)                                                
#define VHALF_M -49

#define k_M 12.5

#define VHALF_H -83

#define k_H -10

#define TAU0_M 0.25

#define TAU1_M 7.0

#define PHI_M -49

#define SIG0_M 29

#define SIG1_M -29

#define TAU0_H 50

#define TAU1_H 121

#define PHI_H -83

#define SIG0_H 10

#define SIG1_H -10



void ChannelKAs_GPe_mouse::update(RNG& rng) 
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

 dyn_var_t taum = TAU0_M + (TAU1_M - TAU0_M) / ( exp( (PHI_M - v)/SIG0_M) + exp( (PHI_M - v)/SIG1_M) );
   dyn_var_t qm = dt  / (taum * 2);
 dyn_var_t tauh = TAU0_H + (TAU1_H - TAU0_H) / ( exp( (PHI_H - v)/SIG0_H) + exp( (PHI_H - v)/SIG1_H) );
   dyn_var_t qh = dt  / (tauh * 2);
   dyn_var_t m_inf = 1.0 / (1 + exp(( VHALF_M - v) / k_M));
   dyn_var_t h_inf = 1.0 / (1 + exp(( VHALF_H - v) / k_H));                               
    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
    { 
    // trick to keep m in [0, 1]
      if (m[i] < 0.0)
      {
        m[i] = 0.0;
      }
      else if (m[i] > 1.0)
      {
        m[i] = 1.0;
      }
      // trick to keep h in [0, 1]
      if (h[i] < 0.0)
      {
        h[i] = 0.0;
      }
      else if (h[i] > 1.0)
      {
        h[i] = 1.0;
      }
    }
 
    g[i] = gbar[i] * m[i] * m[i] * m[i] * m[i] * h[i];
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);  // at time (t+dt/2)
#if defined(WRITE_GATES)                                                  
    if (is_write)
    {           
      (*outFile) << std::fixed << fieldDelimiter << m[i];                 
      (*outFile) << std::fixed << fieldDelimiter << h[i];                 
    }                                                                     
#endif       
}
}
void ChannelKAs_GPe_mouse::initialize(RNG& rng) 
{
//  pthread_once(&once_KAs, ChannelKAs::initialize_others);
  assert(branchData);
  unsigned size = branchData->size;
  assert(V);
  assert(gbar.size() == size);
  assert(V->size() == size);
  // allocate
  if (g.size() != size) g.increaseSizeTo(size);
  if (m.size() != size) m.increaseSizeTo(size);
  if (h.size() != size) h.increaseSizeTo(size);
  if (Iion.size() != size) Iion.increaseSizeTo(size);
  // initialize
  float gbar_default = gbar[0];
  if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on "
                 "Channels KAs (KAp) Param"
              << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
    if (gbar_dists.size() > 0)
    {
      unsigned int j;
      // NOTE: 'n' bins are splitted by (n-1) points
      if (gbar_values.size() - 1 != gbar_dists.size())
      {
        std::cerr << "gbar_values.size = " << gbar_values.size()
                  << "; gbar_dists.size = " << gbar_dists.size() << std::endl;
      }
      assert(gbar_values.size() - 1 == gbar_dists.size());
      for (j = 0; j < gbar_dists.size(); ++j)
      {
        if ((*dimensions)[i]->dist2soma < gbar_dists[j]) break;
      }
      if (j < gbar_values.size())
        gbar[i] = gbar_values[j];
      else
        gbar[i] = gbar_default;
    }
    /*else if (gbar_values.size() == 1) {
gbar[i] = gbar_values[0];
} */
    else if (gbar_branchorders.size() > 0)
    {
      unsigned int j;
      if (gbar_values.size() != gbar_branchorders.size())
      {
        std::cerr << "gbar_values.size = " << gbar_values.size()
                  << "; gbar_branchorders.size = " << gbar_branchorders.size()
                  << std::endl;
      }
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
        gbar[i] = gbar_values[j - 1];
      }
      else if (j < gbar_values.size())
        gbar[i] = gbar_values[j];
      else
        gbar[i] = gbar_default;
    }
    else
    {
      gbar[i] = gbar_default;
    }
  }

  //NOTE: 
  //scale_factor if NOT defined get the value of 1.0
  // if defined; dont use a closed to zero value, i.e. > SMALL
  if (scale_factor < SMALL)
  {
    scale_factor = 1.0;
  }
  else{
    for (unsigned i = 0; i < size; ++i)
    {
      gbar[i] *= scale_factor;
    }
  }
#if defined(WRITE_GATES)                                      
  if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
      _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
  {
    std::ostringstream os;                                    
    std::string fileName = "gates_KAs.txt";                       
    os << fileName << getSimulation().getRank();              
    outFile = new std::ofstream(os.str().c_str());            
    outFile->precision(decimal_places);                       
    (*outFile) << "#Time" << fieldDelimiter << "gates: m, h [, m,h]*"; 
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
   h[i] =1.0 / (1 + exp( (VHALF_H-v) / k_H));                               
   g[i] = gbar[i] * m[i] * m[i] * m[i] * m[i]* h[i];
    
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);  // at time t0+dt/2
#if defined(WRITE_GATES)                                      
    if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
        _segmentDescriptor.getNeuronIndex(branchData->key) == 0)
    {
      (*outFile) << std::fixed << fieldDelimiter << m[i];       
      (*outFile) << std::fixed << fieldDelimiter << h[i];       
    }
#endif                                                        
  }

}

ChannelKAs_GPe_mouse::~ChannelKAs_GPe_mouse() 
{
#if defined(WRITE_GATES)            
  if (outFile) outFile->close();    
#endif       
}

