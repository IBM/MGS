// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
// 
// =============================================================================

#ifndef ChannelSKCompCategory_H
#define ChannelSKCompCategory_H

#include "Lens.h"
#include "CG_ChannelSKCompCategory.h"

#include "CountableModel.h"  //new

class NDPairList;

class ChannelSKCompCategory : public CG_ChannelSKCompCategory,
                              public CountableModel
{
  public:
  ChannelSKCompCategory(Simulation& sim, const std::string& modelName,
                        const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
