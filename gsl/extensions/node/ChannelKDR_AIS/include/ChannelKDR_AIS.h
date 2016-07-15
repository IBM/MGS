#ifndef ChannelKDR_AIS_H
#define ChannelKDR_AIS_H

#include "Lens.h"
#include "CG_ChannelKDR_AIS.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_KDR_AIS == KDR_AIS_TRAUB_1994
#define BASED_TEMPERATURE 23.0 // Celsius
#define Q10 3.0
#endif

#ifndef Q10
#define Q10 3.0 // default
#endif

class ChannelKDR_AIS : public CG_ChannelKDR_AIS
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelKDR_AIS();
   private:
      dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
};

#endif
