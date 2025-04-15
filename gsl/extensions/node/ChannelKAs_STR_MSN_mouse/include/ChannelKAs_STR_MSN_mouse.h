// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#ifndef ChannelKAs_STR_MSN_mouse_H
#define ChannelKAs_STR_MSN_mouse_H

#include "Lens.h"
#include "CG_ChannelKAs_STR_MSN_mouse.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream>

#define BASED_TEMPERATURE 22.0  // Celcius 
#define Q10 2.5                            

class ChannelKAs_STR_MSN_mouse : public CG_ChannelKAs_STR_MSN_mouse
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelKAs_STR_MSN_mouse();
};

#endif
