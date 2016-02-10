#ifndef ChannelKAf_H
#define ChannelKAf_H

#include "Lens.h"
#include "CG_ChannelKAf.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_KAf == KAf_WOLF_2005
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#endif

#ifndef Q10 
#define Q10 3.0 //default
#endif
class ChannelKAf : public CG_ChannelKAf
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelKAf();
  static void initialize_others();  // new
  virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ChannelKAfInAttrPSet* CG_inAttrPset, CG_ChannelKAfOutAttrPSet* CG_outAttrPset);
  private:
  dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);  // new
#if CHANNEL_KAf == KAf_WOLF_2005
	const static dyn_var_t _Vmrange_taum[];
	static dyn_var_t taumKAf[];
  static std::vector<dyn_var_t> Vmrange_taum;
#endif
	int _cptindex;// index of the associated compartment
};

#endif
