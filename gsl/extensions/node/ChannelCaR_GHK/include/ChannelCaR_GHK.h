#ifndef ChannelCaR_GHK_H
#define ChannelCaR_GHK_H

#include "Lens.h"
#include "CG_ChannelCaR_GHK.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_CaR == CaR_GHK_WOLF_2005
#define BASED_TEMPERATURE 22.0 //Celcius
#define Q10 2.3
#elif CHANNEL_CaR == CaR_GHK_TUAN_2017
#define BASED_TEMPERATURE 22.0 //Celcius
#define Q10 2.3
#endif

#ifndef Q10 
#define Q10 3.0 //default
#endif

class ChannelCaR_GHK : public CG_ChannelCaR_GHK
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
  static void initialize_others();
  virtual ~ChannelCaR_GHK();
  #ifdef MICRODOMAIN_CALCIUM
  virtual void setCalciumMicrodomain(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ChannelCaR_GHKInAttrPSet* CG_inAttrPset, CG_ChannelCaR_GHKOutAttrPSet* CG_outAttrPset);
  int _offset; //the offset due to the presence of different Ca2+-microdomain
  #endif
  private:
  dyn_var_t update_current(dyn_var_t v, dyn_var_t cai, int i);
#if CHANNEL_CaR == CaR_GHK_WOLF_2005
  const static dyn_var_t _Vmrange_taum[];
  const static dyn_var_t _Vmrange_tauh[];
  static dyn_var_t taumCaR[];
  static dyn_var_t tauhCaR[];
  static std::vector<dyn_var_t> Vmrange_taum;
  static std::vector<dyn_var_t> Vmrange_tauh;
#endif
};

#endif
