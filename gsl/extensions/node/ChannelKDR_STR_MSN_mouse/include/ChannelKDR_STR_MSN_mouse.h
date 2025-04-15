// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#ifndef ChannelKDR_STR_MSN_mouse_H
#define ChannelKDR_STR_MSN_mouse_H

#include "Lens.h"
#include "CG_ChannelKDR_STR_MSN_mouse.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream> 

#define BASED_TEMPERATURE 22.0  // Celcius                                 
#define Q10 2.92 //To get a phi value equivalent to 5 as used in the model 

class ChannelKDR_STR_MSN_mouse : public CG_ChannelKDR_STR_MSN_mouse
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelKDR_STR_MSN_mouse();
};

#endif
