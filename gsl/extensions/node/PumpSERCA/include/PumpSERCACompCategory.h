// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef PumpSERCACompCategory_H
#define PumpSERCACompCategory_H

#include "Mgs.h"
#include "CG_PumpSERCACompCategory.h"
#include "CountableModel.h"

class NDPairList;

class PumpSERCACompCategory : public CG_PumpSERCACompCategory,
                              public CountableModel
{
  public:
  PumpSERCACompCategory(Simulation& sim, const std::string& modelName,
                        const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
