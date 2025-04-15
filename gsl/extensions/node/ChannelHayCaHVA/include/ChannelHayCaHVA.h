// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHayCaHVA_H
#define ChannelHayCaHVA_H

#include "Lens.h"
#include "CG_ChannelHayCaHVA.h"
#include "rndm.h"

class ChannelHayCaHVA : public CG_ChannelHayCaHVA
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayCaHVA();
   private:
};

#endif
