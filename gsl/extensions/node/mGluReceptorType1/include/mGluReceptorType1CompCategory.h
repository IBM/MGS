// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef mGluReceptorType1CompCategory_H
#define mGluReceptorType1CompCategory_H

#include "Mgs.h"
#include "CG_mGluReceptorType1CompCategory.h"
#include "CountableModel.h"

class NDPairList;

class mGluReceptorType1CompCategory : public CG_mGluReceptorType1CompCategory,
                                 public CountableModel
{
   public:
      mGluReceptorType1CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      //void computeTadj(RNG& rng);
      void count();
};

#endif
