// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ChannelCaLv13_GHKCompCategory_H
#define ChannelCaLv13_GHKCompCategory_H

#include "Mgs.h"
#include "CG_ChannelCaLv13_GHKCompCategory.h"

#include "CountableModel.h"  // new

class NDPairList;

class ChannelCaLv13_GHKCompCategory : public CG_ChannelCaLv13_GHKCompCategory,
                                      public CountableModel
{
  public:
  ChannelCaLv13_GHKCompCategory(Simulation& sim, const std::string& modelName,
                                const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
