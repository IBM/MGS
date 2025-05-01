// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelSchweighoferNat_H
#define ChannelSchweighoferNat_H

#include "Mgs.h"
#include "CG_ChannelSchweighoferNat.h"
#include "rndm.h"

class ChannelSchweighoferNat : public CG_ChannelSchweighoferNat
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelSchweighoferNat();
   private:
};

#endif
