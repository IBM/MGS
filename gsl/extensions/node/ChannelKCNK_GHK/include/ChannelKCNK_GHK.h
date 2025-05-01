// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ChannelKCNK_GHK_H
#define ChannelKCNK_GHK_H

#include "Mgs.h"
#include "CG_ChannelKCNK_GHK.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#define BASED_TEMPERATURE 22.0  // Celcius
#define Q10 2.3

class ChannelKCNK_GHK : public CG_ChannelKCNK_GHK
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelKCNK_GHK();
   private:
      dyn_var_t update_current(dyn_var_t v, dyn_var_t conc_K_i, int i);
};

#endif
