// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ChannelBKalphabetaCompCategory_H
#define ChannelBKalphabetaCompCategory_H

#include "Mgs.h"
#include "CG_ChannelBKalphabetaCompCategory.h"

#include "CountableModel.h" //new

class NDPairList;

class ChannelBKalphabetaCompCategory : public CG_ChannelBKalphabetaCompCategory,
                                   public CountableModel
{
  public:
  ChannelBKalphabetaCompCategory(Simulation& sim, const std::string& modelName,
                             const NDPairList& ndpList);
  void computeE(RNG& rng);
	void count();
};

#endif
