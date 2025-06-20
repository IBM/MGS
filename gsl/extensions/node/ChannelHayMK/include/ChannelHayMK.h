// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHayMK_H
#define ChannelHayMK_H

#include "Mgs.h"
#include "CG_ChannelHayMK.h"
#include "rndm.h"

class ChannelHayMK : public CG_ChannelHayMK
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayMK();
   private:
};

#endif
