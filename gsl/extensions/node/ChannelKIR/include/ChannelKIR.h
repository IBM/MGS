#ifndef ChannelKIR_H
#define ChannelKIR_H

#include "Lens.h"
#include "CG_ChannelKIR.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_KIR  == KIR_WOLF_2005
#define BASED_TEMPERATURE 43.0  // Celcius - in vitro
#define Q10 2.3
#endif

#ifndef Q10 
#define Q10 3.0 //default
#endif
class ChannelKIR : public CG_ChannelKIR
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelKIR();
  static void initialize_others();  // new
  private:
  dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);  // new
#if CHANNEL_KIR == KIR_WOLF_2005
	const static dyn_var_t _Vmrange_taum[];
	static dyn_var_t taumKIR[];
  static std::vector<dyn_var_t> Vmrange_taum;
#endif
};

#endif
