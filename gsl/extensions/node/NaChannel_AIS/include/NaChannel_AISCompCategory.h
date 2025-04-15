// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NaChannel_AISCompCategory_H
#define NaChannel_AISCompCategory_H

#include "Lens.h"
#include "CG_NaChannel_AISCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class NaChannel_AISCompCategory : public CG_NaChannel_AISCompCategory, public CountableModel
{
   public:
      NaChannel_AISCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE_Na(RNG& rng);
      void count();      
};

#endif
