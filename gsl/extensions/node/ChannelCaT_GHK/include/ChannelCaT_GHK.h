#ifndef ChannelCaT_GHK_H
#define ChannelCaT_GHK_H

#include "CG_ChannelCaT_GHK.h"
#include "Lens.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_CaT == CaT_GHK_WOLF_2005
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.3
#elif CHANNEL_CaT == CaT_GHK_TUAN_2017
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.3
#endif

#ifndef Q10
#define Q10 3.0  // default
#endif

class ChannelCaT_GHK : public CG_ChannelCaT_GHK
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelCaT_GHK();
  static void initialize_others();
  #ifdef MICRODOMAIN_CALCIUM
  virtual void setCalciumMicrodomain(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ChannelCaT_GHKInAttrPSet* CG_inAttrPset, CG_ChannelCaT_GHKOutAttrPSet* CG_outAttrPset);
  int _offset; //the offset due to the presence of different Ca2+-microdomain
  #endif

  private:
  dyn_var_t update_current(dyn_var_t v, dyn_var_t cai, int i);
#if CHANNEL_CaT == CaT_GHK_WOLF_2005
  const static dyn_var_t _Vmrange_taum[];
  const static dyn_var_t _Vmrange_tauh[];
  static dyn_var_t taumCaT[];
  static dyn_var_t tauhCaT[];
  static std::vector<dyn_var_t> Vmrange_taum;
  static std::vector<dyn_var_t> Vmrange_tauh;
#endif
};

#endif
