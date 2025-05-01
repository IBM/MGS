// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef LeakyIAFUnitCompCategory_H
#define LeakyIAFUnitCompCategory_H

#include "Mgs.h"
#include "CG_LeakyIAFUnitCompCategory.h"

class NDPairList;

class LeakyIAFUnitCompCategory : public CG_LeakyIAFUnitCompCategory
{
 public:
  LeakyIAFUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
  void initializeShared(RNG& rng);
};

#endif
