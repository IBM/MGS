// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ChannelKCNK_GHKCompCategory_H
#define ChannelKCNK_GHKCompCategory_H

#include "CG_ChannelKCNK_GHKCompCategory.h"
#include "CountableModel.h"
#include "Mgs.h"

class NDPairList;

class ChannelKCNK_GHKCompCategory : public CG_ChannelKCNK_GHKCompCategory,
                                    public CountableModel
{
  public:
  ChannelKCNK_GHKCompCategory(Simulation& sim, const std::string& modelName,
                              const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
