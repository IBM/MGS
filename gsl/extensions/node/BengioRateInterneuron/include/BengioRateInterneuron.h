// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef BengioRateInterneuron_H
#define BengioRateInterneuron_H

#include "Mgs.h"
#include "CG_BengioRateInterneuron.h"
#include "rndm.h"

class BengioRateInterneuron : public CG_BengioRateInterneuron
{
   public:
      void update_U(RNG& rng);
      void update_V(RNG& rng);
      virtual ~BengioRateInterneuron();
};

#endif
