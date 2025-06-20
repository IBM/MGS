// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
// 
// =============================================================================

#ifndef ChannelKv31CompCategory_H
#define ChannelKv31CompCategory_H

#include "Mgs.h"
#include "CG_ChannelKv31CompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKv31CompCategory : public CG_ChannelKv31CompCategory,
                                public CountableModel
{
  public:
  ChannelKv31CompCategory(Simulation& sim, const std::string& modelName,
                          const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
