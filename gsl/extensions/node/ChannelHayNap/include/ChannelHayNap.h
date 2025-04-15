// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHayNap_H
#define ChannelHayNap_H

#include "../../nti/include/MaxComputeOrder.h"

#include "Lens.h"
#include "CG_ChannelHayNap.h"
#include "rndm.h"

class ChannelHayNap : public CG_ChannelHayNap
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayNap();
   private:
};

#endif
