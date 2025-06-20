// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelKDRCompCategory_H
#define ChannelKDRCompCategory_H

#include "Mgs.h"
#include "CG_ChannelKDRCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKDRCompCategory : public CG_ChannelKDRCompCategory,
                               public CountableModel
{
  public:
  ChannelKDRCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
