// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef SwitchInputCompCategory_H
#define SwitchInputCompCategory_H

#include "Mgs.h"
#include "CG_SwitchInputCompCategory.h"

class NDPairList;

class SwitchInputCompCategory : public CG_SwitchInputCompCategory
{
   public:
      SwitchInputCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void updateInputState(RNG& rng);
};

#endif
