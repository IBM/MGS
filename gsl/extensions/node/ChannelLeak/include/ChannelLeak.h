#ifndef ChannelLeak_H
#define ChannelLeak_H

#include "Lens.h"
#include "CG_ChannelLeak.h"
#include "rndm.h"
#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream> 


class ChannelLeak : public CG_ChannelLeak
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelLeak();
};

#endif
