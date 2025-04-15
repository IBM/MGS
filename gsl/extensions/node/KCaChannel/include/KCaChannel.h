// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef KCaChannel_H
#define KCaChannel_H

#include "Lens.h"
#include "CG_KCaChannel.h"
#include "rndm.h"

class KCaChannel : public CG_KCaChannel
{
   public:
      void update(RNG& rng);
      void initializeKCaChannels(RNG& rng);
      virtual ~KCaChannel();
};

#endif
