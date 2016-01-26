#ifndef ChannelNat_H
#define ChannelNat_H

#include "Lens.h"
#include "CG_ChannelNat.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

class ChannelNat : public CG_ChannelNat
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelNat();
		  static void initialize_Q10();
   private:
	  dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
#if CHANNEL_NAT == NAT_WOLF_2005
	static std::vector<dyn_var_t> Vmrange_taum;
	static std::vector<dyn_var_t> Vmrange_tauh;
#endif
};

#endif
