// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#ifndef ChannelLeak_GPe_mouse_H
#define ChannelLeak_GPe_mouse_H

#include "Lens.h"
#include "CG_ChannelLeak_GPe_mouse.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream> 

class ChannelLeak_GPe_mouse : public CG_ChannelLeak_GPe_mouse
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelLeak_GPe_mouse();
};

#endif
