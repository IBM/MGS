// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#ifndef ChannelKRP_STR_MSN_mouse_H
#define ChannelKRP_STR_MSN_mouse_H

#include "Mgs.h"
#include "CG_ChannelKRP_STR_MSN_mouse.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#define BASED_TEMPERATURE 22.0  // Celcius 
#define Q10 2.5                            

class ChannelKRP_STR_MSN_mouse : public CG_ChannelKRP_STR_MSN_mouse
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelKRP_STR_MSN_mouse();
};

#endif
