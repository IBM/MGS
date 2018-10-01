// =================================================================
//
// (C) Copyright 2018 New Jersey Institute of Technology.
//
// =================================================================


#include "Lens.h"
#include "ChannelKDR_STR_MSN_mouse.h"
#include "CG_ChannelKDR_STR_MSN_mouse.h"
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
 
// GOAL: update gates using v(t+dt/2) and gate(t-dt/2)
//   --> output gate(t+dt/2)
//   of second-order accuracy at time (t+dt/2) using trapezoidal rule
void ChannelKDR_STR_MSN_mouse::update(RNG& rng) 
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
 
  for (unsigned i=0; i<branchData->size; ++i) 
  {
    dyn_var_t v=(*V)[i];
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
void ChannelKDR_STR_MSN_mouse::initialize(RNG& rng) 
{
  assert(branchData);
  unsigned size=branchData->size;
  assert(V);
  assert(gbar.size()==size);
  assert (V->size()==size);
  if (g.size()!=size) g.increaseSizeTo(size);
  if (n.size()!=size) n.increaseSizeTo(size);
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
  for (unsigned i=0; i<size; ++i) 
  {
    dyn_var_t v=(*V)[i];
//v is at time (t0)
// so m and h is also at time t0
// however, as they are at steady-state, the value at time (t0+dt/2)
// does not change
    dyn_var_t an = ANC*vtrap((v - ANV), AND);          
    dyn_var_t bn = BNC*exp((v - BNV)/BND);             
    n[i] = an/(an + bn); // steady-state value         
    g[i]=gbar[i] * pow(n[i], 4);
 
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
  }
}

ChannelKDR_STR_MSN_mouse::~ChannelKDR_STR_MSN_mouse() 
{
}

