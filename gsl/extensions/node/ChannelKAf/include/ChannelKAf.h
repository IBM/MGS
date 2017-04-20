#ifndef ChannelKAf_H
#define ChannelKAf_H

#include "CG_ChannelKAf.h"
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
class ChannelKAf : public CG_ChannelKAf
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelKAf();
  static void initialize_others();  // new
  private:
  dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);  // new
#if CHANNEL_KAf == KAf_WOLF_2005
  const static dyn_var_t _Vmrange_taum[];
  static dyn_var_t taumKAf[];
  static std::vector<dyn_var_t> Vmrange_taum;
#endif
};

#endif
