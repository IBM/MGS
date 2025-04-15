// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelCaLVA_H
#define ChannelCaLVA_H

#include "Lens.h"
#include "CG_ChannelCaLVA.h"
#include "rndm.h"
#include "MaxComputeOrder.h"

#if CHANNEL_CaLVA == CaLVA_HAY_2011
#define DUAL_GATE _YES 
#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.3
#endif

class ChannelCaLVA : public CG_ChannelCaLVA
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelCaLVA();
   private:
};

#endif
