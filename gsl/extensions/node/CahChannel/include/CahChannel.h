// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CahChannel_H
#define CahChannel_H

#include "Lens.h"
#include "CG_CahChannel.h"
#include "rndm.h"

class CahChannel : public CG_CahChannel
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~CahChannel();
   private:
};

#endif
