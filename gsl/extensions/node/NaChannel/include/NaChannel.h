// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NaChannel_H
#define NaChannel_H

#include "Lens.h"
#include "CG_NaChannel.h"
#include "rndm.h"

#include "../../nti/include/MaxComputeOrder.h"

class NaChannel : public CG_NaChannel
{
   public:
      void update(RNG& rng);
      void initializeNaChannels(RNG& rng);
      virtual ~NaChannel();
   private:
};

#endif
