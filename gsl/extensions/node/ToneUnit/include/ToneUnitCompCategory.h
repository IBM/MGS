// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ToneUnitCompCategory_H
#define ToneUnitCompCategory_H

#include "Mgs.h"
#include "CG_ToneUnitCompCategory.h"

class NDPairList;

class ToneUnitCompCategory : public CG_ToneUnitCompCategory
{
   public:
      ToneUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
};

#endif
