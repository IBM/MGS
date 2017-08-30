#ifndef ChannelKv31_H
#define ChannelKv31_H

#include "Lens.h"
#include "CG_ChannelKv31.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_Kv31 == Kv31_RETTIG_1992 
#define BASED_TEMPERATURE 23.0  // Celcius
#define Q10 3.0
#endif

class ChannelKv31 : public CG_ChannelKv31
{
  public:
    void update(RNG& rng);
    void initialize(RNG& rng);
    virtual ~ChannelKv31();
  private:
};

#endif
