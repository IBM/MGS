// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NaChannelCompCategory_H
#define NaChannelCompCategory_H

#include "Mgs.h"
#include "CG_NaChannelCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class NaChannelCompCategory : public CG_NaChannelCompCategory, public CountableModel
{
   public:
      NaChannelCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE_Na(RNG& rng);
      void count();      
};

#endif
