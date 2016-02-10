#ifndef ChannelKRP_H
#define ChannelKRP_H

#include "Lens.h"
#include "CG_ChannelKRP.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_KRP == KRP_WOLF_2005
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#endif

#ifndef Q10 
#define Q10 3.0 //default
#endif
class ChannelKRP : public CG_ChannelKRP
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelKRP();
  static void initialize_others();//new
  virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ChannelKRPInAttrPSet* CG_inAttrPset, CG_ChannelKRPOutAttrPSet* CG_outAttrPset);
  private:
  dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);//new
#if CHANNEL_KRP == KRP_WOLF_2005
	const static dyn_var_t _Vmrange_taum[];
	const static dyn_var_t _Vmrange_tauh[];
	static dyn_var_t taumKRP[];
	static dyn_var_t tauhKRP[];
  static std::vector<dyn_var_t> Vmrange_taum;
  static std::vector<dyn_var_t> Vmrange_tauh;
#endif
	int _cptindex;// index of the associated compartment
};

#endif
