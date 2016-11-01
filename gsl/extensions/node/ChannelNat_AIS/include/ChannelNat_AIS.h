#ifndef ChannelNat_AIS_H
#define ChannelNat_AIS_H

#include "Lens.h"
#include "CG_ChannelNat_AIS.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_NAT_AIS == NAT_AIS_TRAUB_1994
#define BASED_TEMPERATURE 25  // Celcius
#define Q10 3.0
#endif

class ChannelNat_AIS : public CG_ChannelNat_AIS
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelNat_AIS();
   private:
      dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
};

#endif
