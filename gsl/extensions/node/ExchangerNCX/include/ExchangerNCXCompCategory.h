// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ExchangerNCXCompCategory_H
#define ExchangerNCXCompCategory_H

#include "Mgs.h"
#include "CG_ExchangerNCXCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ExchangerNCXCompCategory : public CG_ExchangerNCXCompCategory,
                                 public CountableModel
{
  public:
  ExchangerNCXCompCategory(Simulation& sim, const std::string& modelName,
                           const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
