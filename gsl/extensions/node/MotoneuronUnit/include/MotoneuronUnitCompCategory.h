// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef MotoneuronUnitCompCategory_H
#define MotoneuronUnitCompCategory_H

#include "Lens.h"
#include "CG_MotoneuronUnitCompCategory.h"

class NDPairList;

class MotoneuronUnitCompCategory : public CG_MotoneuronUnitCompCategory
{
 public:
  MotoneuronUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
  void initializeShared(RNG& rng);
};

#endif
