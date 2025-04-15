// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CalChannel_H
#define CalChannel_H

#include "Lens.h"
#include "CG_CalChannel.h"
#include "rndm.h"

class CalChannel : public CG_CalChannel
{
   public:
      void update(RNG& rng);
      void initializeCalChannels(RNG& rng);
      virtual ~CalChannel();
   private:
};

#endif
