// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef NVUNodeCompCategory_H
#define NVUNodeCompCategory_H

#include "Mgs.h"
#include "CG_NVUNodeCompCategory.h"

class NDPairList;

class NVUNodeCompCategory : public CG_NVUNodeCompCategory
{
   public:
      NVUNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void paramInitialize(RNG& rng);

};

#endif
