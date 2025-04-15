// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#ifndef ChannelNat_STR_MSN_mouse_H
#define ChannelNat_STR_MSN_mouse_H

#include "Lens.h"
#include "CG_ChannelNat_STR_MSN_mouse.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream>

#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.92 //To get a phi value = 5 at 37oC

#ifndef Q10 
#define Q10 2.3 //default
#endif

class ChannelNat_STR_MSN_mouse : public CG_ChannelNat_STR_MSN_mouse
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelNat_STR_MSN_mouse();
};

#endif
