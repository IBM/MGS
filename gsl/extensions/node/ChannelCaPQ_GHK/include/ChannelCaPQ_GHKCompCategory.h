// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ChannelCaPQ_GHKCompCategory_H
#define ChannelCaPQ_GHKCompCategory_H

#include "Mgs.h"
#include "CG_ChannelCaPQ_GHKCompCategory.h"

#include "CountableModel.h"  // new

class NDPairList;

class ChannelCaPQ_GHKCompCategory : public CG_ChannelCaPQ_GHKCompCategory,
                                    public CountableModel
{
  public:
  ChannelCaPQ_GHKCompCategory(Simulation& sim, const std::string& modelName,
                              const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
