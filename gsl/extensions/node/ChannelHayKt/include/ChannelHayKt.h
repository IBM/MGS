// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHayKt_H
#define ChannelHayKt_H

#include "Mgs.h"
#include "CG_ChannelHayKt.h"
#include "rndm.h"

class ChannelHayKt : public CG_ChannelHayKt
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayKt();
   private:
};

#endif
