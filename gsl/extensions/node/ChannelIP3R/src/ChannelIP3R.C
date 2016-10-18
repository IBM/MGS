#include "Lens.h"
#include "ChannelIP3R.h"
#include "CG_ChannelIP3R.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"
#include "MaxComputeOrder.h"

#define SMALL 1.0E-6
#include <math.h>
#include <pthread.h>
#include <algorithm>

#if CHANNEL_IP3R == IP3R_LI_RINZEL_1994
#define d1  0.13 //[uM]
#define d5  0.08234 //[uM]
#define a2  0.2e-3  // [1/(uM.ms)]
#define d2  1.049 // [uM]
#endif

dyn_var_t ChannelIP3R::vtrap(dyn_var_t x, dyn_var_t y) {
	return(fabs(x/y) < SMALL ? y*(1 - x/y/2) : x/(exp(x/y) - 1));
}

void ChannelIP3R::update(RNG& rng) 
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<branchData->size; ++i) 
  {
    dyn_var_t cai = (*Ca_IC)[i];   //[uM]
    dyn_var_t casr = (*Ca_ER)[i];  //[uM]
    dyn_var_t g=0;
#if CHANNEL_IP3R == IP3R_LI_RINZEL_1994

    mInf[i] = (IP3[i])/(IP3[i]+d1) ;
    nInf[i] = (cai)/(cai + d5);

    dyn_var_t ah = a2 * d2 * (IP3[i] + d1) / (IP3[i] + d3 );
    dyn_var_t bh = a2 * cai;
    dyn_var_t ph = 0.5 * dt * (ah + bh);
    //see Rempe-Chomp (2006)
    h[i] = (dt*ah + h[i]*(1.0 - ph))/(1.0 + ph);

    dyn_var_t VolumeRatio = 0.185; // VolER / VolCyto
    g = VolumeRatio * v_IP3R[i] * pow(mInf[i],3) 
      * pow(nInf[i], 3) * pow(h[i], 3);
		J_Ca[i] = g * (casr - cai); // [uM/ms]
#else
    assert(0);
		//NOT IMPLEMENTED YET
#endif
  }
}

void ChannelIP3R::initialize(RNG& rng) 
{
  assert(branchData);
  unsigned size=branchData->size;
  assert(Ca_ER);
  assert(Ca_ER->size() == size);
  assert(Ca_IC);
  assert(Ca_IC->size() == size);
#if IP3_LOCATION == IP3_DIFFUSIONAL_VAR 
  assert(IP3);
  assert(IP3->size() == size);
#endif
  //assert(gbar.size()==size);
  // allocate
  //if (g.size()!=size) g.increaseSizeTo(size);
  if (h.size()!=size) h.increaseSizeTo(size);
  if (mInf.size()!=size) mInf.increaseSizeTo(size);
  if (nInf.size()!=size) nInf.increaseSizeTo(size);
  if (J_Ca.size()!=size) J_Ca.increaseSizeTo(size);
  // initialize
  SegmentDescriptor segmentDescriptor;

  float v_IP3R_default = v_IP3R[0];
  if (v_IP3R_dists.size() > 0 and v_IP3R_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either v_IP3R_dists or v_IP3R_branchorders on Channels KDR Param"
      << std::endl;
    assert(0);
  }
  for (unsigned i = 0; i < size; ++i)
  {
    //v_IP3R init
    if (v_IP3R_dists.size() > 0) {
      unsigned int j;
      //NOTE: 'n' bins are splitted by (n-1) points
      if (v_IP3R_values.size() - 1 != v_IP3R_dists.size())
      {
        std::cerr << "v_IP3R_values.size = " << v_IP3R_values.size()
          << "; v_IP3R_dists.size = " << v_IP3R_dists.size() << std::endl;
      }
      assert(v_IP3R_values.size() -1 == v_IP3R_dists.size());
      for (j=0; j<v_IP3R_dists.size(); ++j) {
        if ((*dimensions)[i]->dist2soma < v_IP3R_dists[j]) break;
      }
      v_IP3R[i] = v_IP3R_values[j];
    }
    /*else if (v_IP3R_values.size() == 1) {
      v_IP3R[i] = v_IP3R_values[0];
      } */
    else if (v_IP3R_branchorders.size() > 0)
    {
      unsigned int j;
      assert(v_IP3R_values.size() == v_IP3R_branchorders.size());
      SegmentDescriptor segmentDescriptor;
      for (j=0; j<v_IP3R_branchorders.size(); ++j) {
        if (segmentDescriptor.getBranchOrder(branchData->key) == v_IP3R_branchorders[j]) break;
      }
      if (j == v_IP3R_branchorders.size() and v_IP3R_branchorders[j-1] == GlobalNTS::anybranch_at_end)
      {
        v_IP3R[i] = v_IP3R_values[j-1];
      }
      else if (j < v_IP3R_values.size())
        v_IP3R[i] = v_IP3R_values[j];
      else
        v_IP3R[i] = v_IP3R_default;
    }
    else {
      v_IP3R[i] = v_IP3R_default;
    }
  }

  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i=0; i<size; ++i) 
  {
    dyn_var_t cai = (*Ca_IC)[i];   //[uM]
    dyn_var_t casr = (*Ca_ER)[i];  //[uM]
    dyn_var_t g=0;
#if CHANNEL_IP3R == IP3R_LI_RINZEL_1994

    mInf[i] = (IP3[i])/(IP3[i]+d1) ;
    nInf[i] = (cai)/(cai + d5);

    dyn_var_t ah = a2 * d2 * (IP3[i] + d1) / (IP3[i] + d3 );
    dyn_var_t bh = a2 * cai;
    dyn_var_t ph = 0.5 * dt * (ah + bh);
    //see Rempe-Chomp (2006)
    h[i] = (dt*ah + h[i]*(1.0 - ph))/(1.0 + ph);

    dyn_var_t VolumeRatio = 0.185; // VolER / VolCyto
    g = VolumeRatio * v_IP3R[i] * pow(mInf[i],3) 
      * pow(nInf[i], 3) * pow(h[i], 3);
		J_Ca[i] = g * (casr - cai); // [uM/ms]
#endif
  }
}

ChannelIP3R::~ChannelIP3R() 
{
}

