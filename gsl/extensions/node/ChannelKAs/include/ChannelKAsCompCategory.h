// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
// 
// =============================================================================


#ifndef ChannelKAsCompCategory_H
#define ChannelKAsCompCategory_H

#include "Lens.h"
#include "CG_ChannelKAsCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKAsCompCategory : public CG_ChannelKAsCompCategory,
                               public CountableModel
{
  public:
  ChannelKAsCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
