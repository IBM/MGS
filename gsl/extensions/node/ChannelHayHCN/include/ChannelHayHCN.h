// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHayHCN_H
#define ChannelHayHCN_H

#include "Mgs.h"
#include "CG_ChannelHayHCN.h"
#include "rndm.h"
#include "MaxComputeOrder.h"

class ChannelHayHCN : public CG_ChannelHayHCN
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHayHCN();
   private:
};

#endif
