// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHaySK_H
#define ChannelHaySK_H

#include "Lens.h"
#include "CG_ChannelHaySK.h"
#include "rndm.h"

class ChannelHaySK : public CG_ChannelHaySK
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHaySK();
};

#endif
