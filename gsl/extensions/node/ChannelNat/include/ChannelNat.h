#ifndef ChannelNat_H
#define ChannelNat_H

#include "Lens.h"
#include "CG_ChannelNat.h"
#include "rndm.h"

#include "../../nti/include/MaxComputeOrder.h"

class ChannelNat : public CG_ChannelNat
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelNat();
   private:
	  dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
};

#endif
