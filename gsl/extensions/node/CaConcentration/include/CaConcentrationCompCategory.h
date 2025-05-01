// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CaConcentrationCompCategory_H
#define CaConcentrationCompCategory_H

#include "Mgs.h"
#include "CG_CaConcentrationCompCategory.h"
#include "CountableModel.h"

#include "NTSMacros.h"

class NDPairList;

class CaConcentrationCompCategory : public CG_CaConcentrationCompCategory,
                                    public CountableModel
{
  public:
  CaConcentrationCompCategory(Simulation& sim, const std::string& modelName,
                              const NDPairList& ndpList);
  void deriveParameters(RNG& rng);
  void count();
};

#endif
