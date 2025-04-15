// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CaERConcentrationCompCategory_H
#define CaERConcentrationCompCategory_H

#include "Lens.h"
#include "CG_CaERConcentrationCompCategory.h"
#include "CountableModel.h"

#include "NTSMacros.h"

class NDPairList;

class CaERConcentrationCompCategory : public CG_CaERConcentrationCompCategory,
                                      public CountableModel
{
  public:
  CaERConcentrationCompCategory(Simulation& sim, const std::string& modelName,
                                const NDPairList& ndpList);
  void deriveParameters(RNG& rng);
  void count();
};

#endif
