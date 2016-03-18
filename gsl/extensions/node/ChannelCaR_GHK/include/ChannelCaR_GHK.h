#ifndef ChannelCaR_GHK_H
#define ChannelCaR_GHK_H

#include "Lens.h"
#include "CG_ChannelCaR_GHK.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_CaR == CaR_GHK_WOLF_2005
#define BASED_TEMPERATURE 35.0 //Celcius
#define Q10 3.0
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
	 private:
			dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
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
