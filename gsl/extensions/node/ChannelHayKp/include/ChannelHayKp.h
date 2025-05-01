// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHayKp_H
#define ChannelHayKp_H

#include "Mgs.h"
#include "CG_ChannelHayKp.h"
#include "rndm.h"

class ChannelHayKp : public CG_ChannelHayKp
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayKp();
   private:
};

#endif
