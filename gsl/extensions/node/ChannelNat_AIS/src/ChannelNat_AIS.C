// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "ChannelNat_AIS.h"
#include "CG_ChannelNat_AIS.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"
#include "MaxComputeOrder.h"
#include "NumberUtils.h"

#define SMALL 1.0E-6

#if CHANNEL_NAT_AIS == NAT_AIS_TRAUB_1994
//Developed for Gamagenesis in interneurons
//All above conventions for a_m, a_h, b_h remain the same as above except b_m below
//b_m = (BMC * (V - BMV))/(exp((V-BMV)/BMD)-1)
#define Eleak -65.0 //[mV]
#define AMC -0.8
#define AMV (17.2+Eleak)
#define AMD -4.0
#define BMC 0.7
#define BMV (42.2+Eleak)
#define BMD 5.0
#define AHC 0.32
#define AHV (42.0+Eleak)
#define AHD -18.0
#define BHC 10.0
#define BHV (42.0+Eleak)
#define BHD -5.0
#elif CHANNEL_NAT_AIS == NAT_AIS_MSN_TUAN_JAMES_2017
// combine data from Ogata-1990 (striatal neuron)
// Surmeier (1992)
// model is used for simulation dorsal striatal (medium-sized spiny MSN
// cell)
//    at 35.0 Celcius
// minf(Vm) = 1/(1+exp((Vm-Vh)/k))
// hinf(Vm) = 1/(1+exp(Vm-Vh)/k)
//#define Vhalf_act_shift 0.0
//#define Vhalf_inact_shift 0.0
#define OGATA_1990 1
#define FRASER_1993 2
#define SURMEIER_1992 3

#define USE_ACT OGATA_1990

#if USE_ACT == OGATA_1990
#define VHALF_M -25
#define k_M -10.0
#elif USE_ACT == FRASER_1993
#define VHALF_M -28
#define k_M -4.5 // D1-MSN
#else
  NOT AVAILABLE
#endif


#define USE_INACT OGATA_1990

#if USE_INACT == OGATA_1990
#define VHALF_H -62
#define k_H 6
#elif USE_INACT == FRASER_1993
#define Vhshift 8
#define VHALF_H (-62.9 + Vhshift)
#define k_H 6
#elif USE_INACT == SURMEIER_1992
#define VHALF_H (-54.6)
#define k_H 5.4   // D2-MSN
//#define k_H 10.2   // D1-MSN
#else
  NOT AVAILABLE
#endif

#define LOOKUP_TAUM_LENGTH 16  // size of the below array
//#define scale_tau_m 1.2
//#define scale_tau_m 1.0
//#define scale_tau_h 1.2
const dyn_var_t ChannelNat_AIS::_Vmrange_taum[] = {-100, -90, -80, -70, -60, -50, -40, -30,
                                   -20,  -10, 0,   10,  20,  30,  40,  50};
// NOTE:
// if (-100+(-90))/2 >= Vm               : tau_m = taumNat[1st-element]
// if (-100+(-90))/2 < Vm < (-90+(-80))/2: tau_m = taumNat[2nd-element]
//...
#if 1
dyn_var_t ChannelNat_AIS::taumNat[] = {0.060, 0.060, 0.070, 0.090, 0.110, 0.130, 0.200, 0.320,
                       0.160, 0.150, 0.120, 0.080, 0.060, 0.060, 0.060, 0.060};
#else
dyn_var_t ChannelNat_AIS::taumNat[] = {0.316, 0.316, 0.351, 0.447, 0.556, 0.355, 0.240, 0.159,
                       0.105, 0.087, 0.085, 0.081, 0.083, 0.083, 0.083, 0.083};
#endif
#define LOOKUP_TAUH_LENGTH 16  // size of the below array
// dyn_var_t _Vmrange_tauh[] = _Vmrange_taum;
const dyn_var_t ChannelNat_AIS::_Vmrange_tauh[] = {-100, -90, -80, -70, -60, -50, -40, -30,
                                   -20, -15, -10, -5, 0, 5,  10,  20};
#if 0
//Nav1.1
dyn_var_t ChannelNat_AIS::tauhNat[] = {5.9196, 5.9196, 5.9197, 6.9103, 8.2985, 3.9111, 1.4907, 1.2596,
                       0.8101, 0.5267, 0.4673, 0.3370, 0.3204, 0.3177, 0.3151, 0.3142};
#else
//Nav1.6 - prevalence in AIS
dyn_var_t ChannelNat_AIS::tauhNat[] = {5.9196, 5.9196, 5.9197, 6.9103, 8.2985, 3.9111, 1.4907, 1.2596,
                       1.410, 1.127, 0.967, 0.857, 0.764, 0.700, 0.627, 0.600};
#endif
std::vector<dyn_var_t> ChannelNat_AIS::Vmrange_taum;
std::vector<dyn_var_t> ChannelNat_AIS::Vmrange_tauh;

#endif

#ifndef scale_tau_m
#define scale_tau_m 1.0
#endif
#ifndef scale_tau_h
#define scale_tau_h 1.0
#endif


void ChannelNat_AIS::update(RNG& rng)
{
  dyn_var_t dt = *(getSharedMembers().deltaT);
  for (unsigned i = 0; i < branchData->size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_NAT_AIS == NAT_AIS_TRAUB_1994
    {
    dyn_var_t am = AMC * vtrap((v - AMV), AMD);
    //dyn_var_t bm = (BMC * (v-BMV)) /  (exp((v - BMV) / BMD) - 1);
    dyn_var_t bm = BMC * vtrap((v - BMV), BMD);
    dyn_var_t ah = AHC * exp((v - AHV) / AHD);
    dyn_var_t bh = BHC / (1.0 + exp((v - BHV) / BHD));
    // Traub Models do not have temperature dependence and hence Tadj is not used
    dyn_var_t pm = 0.5 * dt * (am + bm) ;
    m[i] = (dt * am  + m[i] * (1.0 - pm)) / (1.0 + pm);
    dyn_var_t ph = 0.5 * dt * (ah + bh) ;
    h[i] = (dt * ah  + h[i] * (1.0 - ph)) / (1.0 + ph);
    }
#elif CHANNEL_NAT_AIS == NAT_AIS_MSN_TUAN_JAMES_2017
    {
    // NOTE: Some models use m_inf and tau_m to estimate m
    // tau_m in the lookup table
    std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
    int index = low - Vmrange_taum.begin();
    assert(index>=0);
    //assert(Vmrange_taum[index-1] <= v and v <= Vmrange_taum[index]);
    dyn_var_t taum;
    if (index == 0)
      taum = taumNat[0];
    else if (index < LOOKUP_TAUM_LENGTH)
     taum = linear_interp(Vmrange_taum[index-1], taumNat[index-1],
        Vmrange_taum[index], taumNat[index], v);
    else //assume saturation in taum when Vm > max-value
     taum = taumNat[index-1];
    taum *= scale_tau_m;
    //-->tau_m[i] = taumNat[index];
    // NOTE: dyn_var_t qm = dt * getSharedMembers().Tadj / (tau_m[i] * 2);
    //dyn_var_t qm = dt * getSharedMembers().Tadj / (taumNat[index] * 2);
    dyn_var_t qm = dt * getSharedMembers().Tadj / (taum * 2);
    /* no need to search as they both use the same Vmrange
     * IF NOT< make sure you add this code
    std::vector<dyn_var_t>::iterator low= std::lower_bound(Vmrange_tauh.begin(),
    Vmrange_tauh.end(), v);
    int index = low-Vmrange_tauh.begin();
    */
    dyn_var_t tauh;
    if (index == 0)
      tauh = tauhNat[0];
    else if (index < LOOKUP_TAUH_LENGTH)
      tauh = linear_interp(Vmrange_tauh[index-1], tauhNat[index-1],
        Vmrange_tauh[index], tauhNat[index], v);
    else //assume saturation in taum when Vm > max-value
     tauh = tauhNat[index-1];
    taum *= scale_tau_h;
    //dyn_var_t qh = dt * getSharedMembers().Tadj / (tauhNat[index] * 2);
    dyn_var_t qh = dt * getSharedMembers().Tadj / (tauh * 2);

    //dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M - Vhalf_act_shift[i]) / k_M));
    //dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H - Vhalf_inact_shift[i]) / k_H));
    dyn_var_t m_inf = 1.0 / (1 + exp((v - VHALF_M - Vhalf_act_shift) / k_M));
    dyn_var_t h_inf = 1.0 / (1 + exp((v - VHALF_H - Vhalf_inact_shift) / k_H));

    m[i] = (2 * m_inf * qm - m[i] * (qm - 1)) / (qm + 1);
    h[i] = (2 * h_inf * qh - h[i] * (qh - 1)) / (qh + 1);
    }
#endif
    {//keep range [0..1]
    // trick to keep m in [0, 1]
    if (m[i] < 0.0) { m[i] = 0.0; }
    else if (m[i] > 1.0) { m[i] = 1.0; }
    // trick to keep h in [0, 1]
    if (h[i] < 0.0) { h[i] = 0.0; }
    else if (h[i] > 1.0) { h[i] = 1.0; }
    }
#if CHANNEL_NAT_AIS == NAT_AIS_TRAUB_1994
    g[i] = gbar[i] * m[i] *  m[i] * m[i] * h[i];
    Iion[i] = g[i] * (v - getSharedMembers().E_Na[0]);
#endif
  }
}


void ChannelNat_AIS::initialize(RNG& rng)
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
  if (Iion.size()!=size) Iion.increaseSizeTo(size);
  // initialize
  SegmentDescriptor segmentDescriptor;
  float gbar_default = gbar[0];
  if (gbar_dists.size() > 0 and gbar_branchorders.size() > 0)
  {
    std::cerr << "ERROR: Use either gbar_dists or gbar_branchorders on Channels Nat_AIS Param"
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
  for (unsigned i = 0; i < size; ++i)
  {
    dyn_var_t v = (*V)[i];
#if CHANNEL_NAT_AIS == NAT_AIS_TRAUB_1994
    dyn_var_t am = AMC * vtrap((v - AMV), AMD);
    //dyn_var_t bm = (BMC * (v-BMV)) /  (exp((v - BMV) / BMD) - 1);
    dyn_var_t bm = BMC * vtrap((v - BMV), BMD);
    dyn_var_t ah = AHC * exp((v - AHV) / AHD);
    dyn_var_t bh = BHC / (1.0 + exp((v - BHV) / BHD));
    m[i] = am / (am + bm);  // steady-state value
    h[i] = ah / (ah + bh);
    g[i] = gbar[i] * m[i] *  m[i] * m[i] * h[i];
#elif CHANNEL_NAT_AIS == NAT_AIS_MSN_TUAN_JAMES_2017
    m[i] = 1.0 / (1 + exp((v - VHALF_M - Vhalf_act_shift) / k_M));
    h[i] = 1.0 / (1 + exp((v - VHALF_H - Vhalf_inact_shift) / k_H));
    g[i] = gbar[i] * m[i] * m[i] * m[i] * h[i]; // at time (t+dt/2) -
#else
    NOT IMPLEMENTED;
#endif
    Iion[i] = g[i] * (v - getSharedMembers().E_Na[0]);

  }
}

void ChannelNat_AIS::initialize_others()
{
#if CHANNEL_NAT_AIS == NAT_AIS_MSN_TUAN_JAMES_2017
  {
    //NOTE:
    //  0 <= i < size-1: _Vmrange_tauh[i] << Vm << _Vmrange_tauh[i+1]
    //  or
    //  i = size-1: _Vmrange_tauh[i] << Vm
    //  then :  taum[i]
    std::vector<dyn_var_t> tmp(_Vmrange_taum,
                               _Vmrange_taum + LOOKUP_TAUM_LENGTH);
    //assert((sizeof(taumNat) / sizeof(taumNat[0])) == tmp.size());
		//Vmrange_taum.resize(tmp.size()-2);
    //for (unsigned long i = 1; i < tmp.size() - 1; i++)
    //  Vmrange_taum[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
    Vmrange_taum = tmp;
  }
  {
    std::vector<dyn_var_t> tmp(_Vmrange_tauh,
                               _Vmrange_tauh + LOOKUP_TAUH_LENGTH);
    assert(sizeof(tauhNat) / sizeof(tauhNat[0]) == tmp.size());
		//Vmrange_tauh.resize(tmp.size()-2);
    //for (unsigned long i = 1; i < tmp.size() - 1; i++)
    //  Vmrange_tauh[i - 1] = (tmp[i - 1] + tmp[i + 1]) / 2;
    Vmrange_tauh = tmp;
  }
#endif
}


ChannelNat_AIS::~ChannelNat_AIS()
{
}

