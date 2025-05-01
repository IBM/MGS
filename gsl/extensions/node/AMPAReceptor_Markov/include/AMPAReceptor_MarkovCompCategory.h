// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef AMPAReceptor_MarkovCompCategory_H
#define AMPAReceptor_MarkovCompCategory_H

#include "Mgs.h"
#include "CG_AMPAReceptor_MarkovCompCategory.h"
#include "CountableModel.h"  //new

class NDPairList;

class AMPAReceptor_MarkovCompCategory
    : public CG_AMPAReceptor_MarkovCompCategory,
      public CountableModel
{
  public:
  AMPAReceptor_MarkovCompCategory(Simulation& sim, const std::string& modelName,
                                  const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
