// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef MotoneuronUnit_H
#define MotoneuronUnit_H

#include "Mgs.h"
#include "CG_MotoneuronUnit.h"
#include "rndm.h"

class MotoneuronUnit : public CG_MotoneuronUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual ~MotoneuronUnit();
};

#endif
