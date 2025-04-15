// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GABAAReceptorCompCategory_H
#define GABAAReceptorCompCategory_H

#include "Lens.h"
#include "CG_GABAAReceptorCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class GABAAReceptorCompCategory : public CG_GABAAReceptorCompCategory,
                                  public CountableModel
{
  public:
  GABAAReceptorCompCategory(Simulation& sim, const std::string& modelName,
                            const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
