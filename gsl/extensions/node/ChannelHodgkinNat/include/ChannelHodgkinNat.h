// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHodgkinNat_H
#define ChannelHodgkinNat_H

#include "Mgs.h"
#include "CG_ChannelHodgkinNat.h"
#include "rndm.h"

class ChannelHodgkinNat : public CG_ChannelHodgkinNat
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHodgkinNat();
   private:
};

#endif
