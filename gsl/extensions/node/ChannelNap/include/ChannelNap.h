#ifndef ChannelNap_H
#define ChannelNap_H

#include "Lens.h"
#include "CG_ChannelNap.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_NAP == NAP_WOLF_2005
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#elif CHANNEL_NAP == NAP_MAGISTRETTI_1999
//NOTE: Hay et al. (2011) also use this model
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.3
#endif

#ifndef Q10 
#define Q10 3.0 //default
#endif
class ChannelNap : public CG_ChannelNap
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelNap();

  static void initialize_others();//new

  private:
  dyn_var_t vtrap(dyn_var_t x, dyn_var_t y); //new
#if CHANNEL_NAP == NAP_WOLF_2005
	const static dyn_var_t _Vmrange_tauh[];
	static dyn_var_t tauhNap[];
	static std::vector<dyn_var_t> Vmrange_tauh;
#endif
};

#endif
