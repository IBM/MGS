// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef SupervisorNode_H
#define SupervisorNode_H

#include "Mgs.h"
#include "CG_SupervisorNode.h"
#include "rndm.h"

class SupervisorNode : public CG_SupervisorNode
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual ~SupervisorNode();
};

#endif
