#ifndef ChannelKAf_KChIP_H
#define ChannelKAf_KChIP_H

#include "CG_ChannelKAf_KChIP.h"
#include "Lens.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_KAf == KAf_TRAUB_1994  // There is no temperature dependence
#define BASED_TEMPERATURE 23.0     // Celcius
#define Q10 3.0
#elif CHANNEL_KAf == KAf_KORNGREEN_SAKMANN_2000
#define BASED_TEMPERATURE 21.0  // Celcius
#define Q10 2.3
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
  virtual void setCalciumMicrodomain(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ChannelKAf_KChIPInAttrPSet* CG_inAttrPset, CG_ChannelKAf_KChIPOutAttrPSet* CG_outAttrPset);
  int _offset; //the offset due to the presence of different Ca2+-microdomain
#endif
  float KChIP_Cav_on_conductance(dyn_var_t cai){
    //calculate the effect of KChIP_Cav3 channels on conductance
    // return gbarAdj
    const float Kd = 5.0; //uM
    const float Vmax = 10; // scaling factor
    //return (1 + cai / (cai + Kd));
    //const int n = 4; // 4 EF-hand binding sites
    const int n = 2; // assume 2binding is enough to activate the 4 EF-hand binding sites
    return (1 + Vmax * pow(cai,n) / (pow(cai,n) + pow(Kd,n)));
  }
  private:
  dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);  // new
#if CHANNEL_KAf == KAf_WOLF_2005
  const static dyn_var_t _Vmrange_taum[];
  static dyn_var_t taumKAf[];
  static std::vector<dyn_var_t> Vmrange_taum;
#endif
};

#endif
