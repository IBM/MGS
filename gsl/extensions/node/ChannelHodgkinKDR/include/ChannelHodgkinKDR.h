// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHodgkinKDR_H
#define ChannelHodgkinKDR_H

#include "Mgs.h"
#include "CG_ChannelHodgkinKDR.h"
#include "rndm.h"

class ChannelHodgkinKDR : public CG_ChannelHodgkinKDR
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelHodgkinKDR();
   private:
};

#endif
