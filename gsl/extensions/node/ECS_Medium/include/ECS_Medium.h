// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ECS_Medium_H
#define ECS_Medium_H

#include "Mgs.h"
#include "CG_ECS_Medium.h"
#include "rndm.h"

class ECS_Medium : public CG_ECS_Medium
{
   public:
      void initParams(RNG& rng);
      void update(RNG& rng);
      void copy(RNG& rng);
      void finalize(RNG& rng);
      virtual ~ECS_Medium();
};

#endif
