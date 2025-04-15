// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHayKv31_H
#define ChannelHayKv31_H

#include "Lens.h"
#include "CG_ChannelHayKv31.h"
#include "rndm.h"

class ChannelHayKv31 : public CG_ChannelHayKv31
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayKv31();
   private:
};

#endif
