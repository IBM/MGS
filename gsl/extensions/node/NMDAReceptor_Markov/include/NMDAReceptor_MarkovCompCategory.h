// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef NMDAReceptor_MarkovCompCategory_H
#define NMDAReceptor_MarkovCompCategory_H

#include "CG_NMDAReceptor_MarkovCompCategory.h"
#include "CountableModel.h"
#include "Mgs.h"

class NDPairList;

class NMDAReceptor_MarkovCompCategory
    : public CG_NMDAReceptor_MarkovCompCategory,
      public CountableModel
{
  public:
  NMDAReceptor_MarkovCompCategory(Simulation& sim, const std::string& modelName,
                                  const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
