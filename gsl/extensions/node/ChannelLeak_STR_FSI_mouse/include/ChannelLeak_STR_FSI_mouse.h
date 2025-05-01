// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#ifndef ChannelLeak_STR_FSI_mouse_H
#define ChannelLeak_STR_FSI_mouse_H

#include "Mgs.h"
#include "CG_ChannelLeak_STR_FSI_mouse.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream> 

class ChannelLeak_STR_FSI_mouse : public CG_ChannelLeak_STR_FSI_mouse
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelLeak_STR_FSI_mouse();
};

#endif
