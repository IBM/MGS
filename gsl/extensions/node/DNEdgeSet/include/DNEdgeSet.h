// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef DNEdgeSet_H
#define DNEdgeSet_H

#include "Mgs.h"
#include "CG_DNEdgeSet.h"
#include "rndm.h"
#include "TransferFunction.h"

class DNEdgeSet : public CG_DNEdgeSet
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual ~DNEdgeSet();
      TransferFunction transferFunction;
};

#endif
