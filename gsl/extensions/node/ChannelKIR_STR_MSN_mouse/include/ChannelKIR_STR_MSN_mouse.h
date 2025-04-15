// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#ifndef ChannelKIR_STR_MSN_mouse_H
#define ChannelKIR_STR_MSN_mouse_H

#include "Lens.h"
#include "CG_ChannelKIR_STR_MSN_mouse.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#define BASED_TEMPERATURE 22.0  // Celcius - in vitro  
#define Q10 2.5                                        


class ChannelKIR_STR_MSN_mouse : public CG_ChannelKIR_STR_MSN_mouse
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelKIR_STR_MSN_mouse();
};

#endif
