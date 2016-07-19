#ifndef ChannelBK_H
#define ChannelBK_H

#include "Lens.h"
#include "CG_ChannelBK.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_BK == BK_TRAUB_1994
#define BASED_TEMPERATURE 23.0 // Celsius
#define Q10 3.0
#endif

#ifndef Q10
#define Q10 3.0 // default
#endif

class ChannelBK : public CG_ChannelBK
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelBK();
};

#endif
