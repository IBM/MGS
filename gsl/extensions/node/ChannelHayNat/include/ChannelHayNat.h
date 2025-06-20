// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHayNat_H
#define ChannelHayNat_H

#include "Mgs.h"
#include "CG_ChannelHayNat.h"
#include "rndm.h"

#include "../../nti/include/MaxComputeOrder.h"

class ChannelHayNat : public CG_ChannelHayNat
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayNat();
   private:
};

#endif
