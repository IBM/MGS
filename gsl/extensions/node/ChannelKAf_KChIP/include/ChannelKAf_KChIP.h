// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ChannelKAf_KChIP_H
#define ChannelKAf_KChIP_H

#include "CG_ChannelKAf_KChIP.h"
#include "Mgs.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_KAf == KAf_TRAUB_1994  // There is no temperature dependence
#define BASED_TEMPERATURE 23.0     // Celcius
#define Q10 3.0

#elif CHANNEL_KAf == KAf_KORNGREEN_SAKMANN_2000
#define BASED_TEMPERATURE 21.0  // Celcius
#define Q10 2.3

#elif CHANNEL_KAf == KAf_MAHON_2000           
#define BASED_TEMPERATURE 22.0 // Celcius     
#define Q10 2.5                               

#elif CHANNEL_KAf == KAf_WOLF_2005
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.3

#elif CHANNEL_KAf == KAf_EVANS_2012
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.3
#endif

#ifndef Q10
#define Q10 3.0  // default
#endif

class ChannelKAf_KChIP : public CG_ChannelKAf_KChIP
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelKAf_KChIP();
  static void initialize_others();  // new
#ifdef MICRODOMAIN_CALCIUM
  virtual void setCalciumMicrodomain(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ChannelKAf_KChIPInAttrPSet* CG_inAttrPset, CG_ChannelKAf_KChIPOutAttrPSet* CG_outAttrPset);
  int _offset; //the offset due to the presence of different Ca2+-microdomain
  void KChIP_modulation(dyn_var_t v, unsigned i, 
      dyn_var_t& gbarAdj, dyn_var_t& vm_shift, dyn_var_t& vm_slope_shift);
  float KChIP_Cav_on_conductance(dyn_var_t cai);
#endif
  private:
#if CHANNEL_KAf == KAf_WOLF_2005
  const static dyn_var_t _Vmrange_taum[];
  static dyn_var_t taumKAf[];
  static std::vector<dyn_var_t> Vmrange_taum;
#endif
};

#endif
