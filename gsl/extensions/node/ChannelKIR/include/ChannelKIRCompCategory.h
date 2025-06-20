// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ChannelKIRCompCategory_H
#define ChannelKIRCompCategory_H

#include "Mgs.h"
#include "CG_ChannelKIRCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKIRCompCategory : public CG_ChannelKIRCompCategory,
                               public CountableModel
{
  public:
  ChannelKIRCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
