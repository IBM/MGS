// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelSchweighoferCah_H
#define ChannelSchweighoferCah_H

#include "Lens.h"
#include "CG_ChannelSchweighoferCah.h"
#include "rndm.h"

class ChannelSchweighoferCah : public CG_ChannelSchweighoferCah
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelSchweighoferCah();
   private:
};

#endif
