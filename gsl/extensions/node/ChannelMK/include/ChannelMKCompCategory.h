// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelMKCompCategory_H
#define ChannelMKCompCategory_H

#include "Lens.h"
#include "CG_ChannelMKCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelMKCompCategory : public CG_ChannelMKCompCategory,
                              public CountableModel
{
  public:
  ChannelMKCompCategory(Simulation& sim, const std::string& modelName,
                        const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
