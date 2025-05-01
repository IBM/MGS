// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef CaExtrusionCompCategory_H
#define CaExtrusionCompCategory_H

#include "Mgs.h"
#include "CG_CaExtrusionCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class CaExtrusionCompCategory : public CG_CaExtrusionCompCategory,
                                public CountableModel
{
  public:
  CaExtrusionCompCategory(Simulation& sim, const std::string& modelName,
                          const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
