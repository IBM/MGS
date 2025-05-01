// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ChannelKDR_AIS_H
#define ChannelKDR_AIS_H

#include "Mgs.h"
#include "CG_ChannelKDR_AIS.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if CHANNEL_KDR_AIS == KDR_AIS_TRAUB_1994 || \
    CHANNEL_KDR_AIS == KDR_AIS_TRAUB_1995 //not being used for these models
#define BASED_TEMPERATURE 23.0 // Celsius
#define Q10 3.0
#endif

#ifndef Q10
#define Q10 3.0 // default
#endif

class ChannelKDR_AIS : public CG_ChannelKDR_AIS
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelKDR_AIS();
   private:
};

#endif
