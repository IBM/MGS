// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef AMPAReceptorCompCategory_H
#define AMPAReceptorCompCategory_H

#include "Lens.h"
#include "CG_AMPAReceptorCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class AMPAReceptorCompCategory : public CG_AMPAReceptorCompCategory,
                                 public CountableModel
{
  public:
  AMPAReceptorCompCategory(Simulation& sim, const std::string& modelName,
                           const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
