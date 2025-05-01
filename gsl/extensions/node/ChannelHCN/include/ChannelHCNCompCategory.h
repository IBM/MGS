// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHCNCompCategory_H
#define ChannelHCNCompCategory_H

#include "Mgs.h"
#include "CG_ChannelHCNCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelHCNCompCategory : public CG_ChannelHCNCompCategory,
                               public CountableModel
{
  public:
  ChannelHCNCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
