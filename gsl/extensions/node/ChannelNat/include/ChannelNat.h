#ifndef ChannelNat_H
#define ChannelNat_H

#include "Lens.h"
#include "CG_ChannelNat.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_NAT == NAT_HODGKINHUXLEY_1952
#define BASED_TEMPERATURE 6.3  // Celcius
#define Q10 3.0
#elif CHANNEL_NAT == NAT_SCHWEIGHOFER_1999
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#elif CHANNEL_NAT == NAT_WOLF_2005
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#elif CHANNEL_NAT == NAT_HAY_2011
#define BASED_TEMPERATURE 23  // Celcius
#define Q10 2.3
#elif CHANNEL_NAT == NAT_RUSH_RINZEL_1994
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#endif

#ifndef Q10 
#define Q10 3.0 //default
#endif

class ChannelNat : public CG_ChannelNat
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelNat();
  static void initialize_others();
  virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ChannelNatInAttrPSet* CG_inAttrPset, CG_ChannelNatOutAttrPSet* CG_outAttrPset);

  private:
  dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
#if CHANNEL_NAT == NAT_WOLF_2005
	const static dyn_var_t _Vmrange_taum[];
	const static dyn_var_t _Vmrange_tauh[];
	static dyn_var_t taumNaf[];
	static dyn_var_t tauhNaf[];
  static std::vector<dyn_var_t> Vmrange_taum;
  static std::vector<dyn_var_t> Vmrange_tauh;
#endif
	int _cptindex;// index of the associated compartment
};

#endif
