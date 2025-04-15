// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelSchweighoferNatCompCategory_H
#define ChannelSchweighoferNatCompCategory_H

#include "Lens.h"
#include "CG_ChannelSchweighoferNatCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelSchweighoferNatCompCategory : public CG_ChannelSchweighoferNatCompCategory, public CountableModel
{
   public:
      ChannelSchweighoferNatCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();      
};

#endif
