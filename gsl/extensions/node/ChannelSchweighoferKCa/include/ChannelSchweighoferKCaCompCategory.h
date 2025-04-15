// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelSchweighoferKCaCompCategory_H
#define ChannelSchweighoferKCaCompCategory_H

#include "Lens.h"
#include "CG_ChannelSchweighoferKCaCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelSchweighoferKCaCompCategory : public CG_ChannelSchweighoferKCaCompCategory, public CountableModel
{
   public:
      ChannelSchweighoferKCaCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();      
};

#endif
