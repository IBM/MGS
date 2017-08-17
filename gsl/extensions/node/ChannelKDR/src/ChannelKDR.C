// =================================================================
// Licensed Materials - Property of IBM
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "ChannelKDR.h"
#include "CG_ChannelKDR.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"
#include "MaxComputeOrder.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

// NOTE: vtrap(x,y) = x/(exp(x/y)-1)
// a_m  = AMC*(V - AMV)/( exp( (V - AMV)/AMD ) - 1.0 )
// b_m  = BMC * exp( (V - BMV)/BMD )
// a_h  = AHC * exp( (V - AHV)/AHD )
// b_h  = BHC / (exp( (V - BHV)/BHD ) + 1.0)
#if CHANNEL_KDR == KDR_HODGKIN_HUXLEY_1952
// data measured from squid giant axon
#define ANC 0.01
#define ANV -55.0
#define AND 10.0
#define BNC 0.125
#define BNV -65.0
#define BND 80.0
#elif CHANNEL_KDR == KDR_SCHWEIGHOFER_1999
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
#define Eleak -65.0 //mV
#define ANC 0.016
#define ANV (35.1+Eleak)
#define AND 5
#define BNC 0.25
#define BNV (20+Eleak)
#define BND 40

#elif CHANNEL_KDR == KDR_WANG_BUSZAKI_1996
// Equations from paper Wang_Buzsaki_1996 
// IK = gK * n^4 (V-E)
// NOTE: vtrap(x,y) = x/(exp(x/y)-1)
#define Vshift 0 
#define ANC -0.01                                                                  
#define ANV (-34.0+Vshift) 
#define AND -10.0                                                                  
#define BNC 0.125                                                                  
#define BNV (-44.0+Vshift)
#define BND -80.0 

#elif CHANNEL_KDR == KDR_MAHON_2000                                                
// Equations from paper Wang_Buzsaki_1996, half activation voltages shifted by 7mV 
#define Vshift 7.0 
#define ANC -0.01                                                                  
#define ANV (-34.0+Vshift) 
#define AND -10.0                                                                  
#define BNC 0.125                                                                  
#define BNV (-44.0+Vshift)
#define BND -80.0 
                                                                                   
#endif

// NOTE: vtrap(x,y) = x/(exp(x/y)-1)
dyn_var_t ChannelKDR::vtrap(dyn_var_t x, dyn_var_t y) {
	return(fabs(x/y) < SMALL ? y*(1 - x/y/2) : x/(exp(x/y) - 1));
}

// GOAL: update gates using v(t+dt/2) and gate(t-dt/2)
//   --> output gate(t+dt/2)
//   of second-order accuracy at time (t+dt/2) using trapezoidal rule
void ChannelKDR::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) 
  {
    dyn_var_t v=(*V)[i];
#if CHANNEL_KDR == KDR_HODGKIN_HUXLEY_1952  
    dyn_var_t an = ANC*vtrap(-(v - ANV), AND);
    dyn_var_t bn = BNC*exp(-(v - BNV)/BND);
    // see Rempe-Chomp (2006)
    dyn_var_t pn = 0.5 * dt * (an + bn) * getSharedMembers().Tadj;
    n[i] = (dt*an*getSharedMembers().Tadj + n[i]*(1.0 - pn))/(1.0 + pn);

    g[i] = gbar[i]*n[i]*n[i]*n[i]*n[i];
    //KDR = gKDR * n^4 * (Vm - Erev)
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);

#elif    CHANNEL_KDR == KDR_SCHWEIGHOFER_1999
    dyn_var_t an = ANC * vtrap(-(v - ANV), AND);
    dyn_var_t bn = BNC * (exp(-(v-BNV)/BND));
    dyn_var_t n_inf = an / (an + bn);
    dyn_var_t qn = 0.5 * dt * (an + bn) * getSharedMembers().Tadj;
    n[i] = (2 * n_inf * qn - n[i] * (qn - 1)) / (qn + 1);
    //KDR = gKDR * n^4 * (Vm - Erev)
    g[i] = gbar[i]*n[i]*n[i]*n[i]*n[i];
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
                                                                           
    g[i] = gbar[i]*n[i]*n[i]*n[i]*n[i]; 
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]); // at time (t+dt/2) 

#else
    NOT IMPLEMENTED YET
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
#if CHANNEL_KDR == KDR_HODGKIN_HUXLEY_1952 
    dyn_var_t an = ANC*vtrap(-(v - ANV), AND);
    dyn_var_t bn = BNC*exp(-(v - BNV)/BND);
    n[i] = an/(an + bn); // steady-state value
    g[i]=gbar[i]*n[i]*n[i]*n[i]*n[i];
#elif CHANNEL_KDR == KDR_SCHWEIGHOFER_1999
    dyn_var_t an = ANC*vtrap(-(v - ANV), AND);
    dyn_var_t bn = BNC*exp(-(v - BNV)/BND);
    n[i] = an/(an + bn); // steady-state value
    g[i]=gbar[i]*n[i]*n[i]*n[i]*n[i];
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
    g[i]=gbar[i]*n[i]*n[i]*n[i]*n[i];                  

#endif
    Iion[i] = g[i] * (v - getSharedMembers().E_K[0]);
  }
}

ChannelKDR::~ChannelKDR()
{
}

