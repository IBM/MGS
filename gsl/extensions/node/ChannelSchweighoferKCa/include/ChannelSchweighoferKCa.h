// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelSchweighoferKCa_H
#define ChannelSchweighoferKCa_H

#include "Lens.h"
#include "CG_ChannelSchweighoferKCa.h"
#include "rndm.h"

class ChannelSchweighoferKCa : public CG_ChannelSchweighoferKCa
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelSchweighoferKCa();
};

#endif
