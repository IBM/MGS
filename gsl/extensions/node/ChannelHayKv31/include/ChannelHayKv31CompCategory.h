// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHayKv31CompCategory_H
#define ChannelHayKv31CompCategory_H

#include "Lens.h"
#include "CG_ChannelHayKv31CompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelHayKv31CompCategory : public CG_ChannelHayKv31CompCategory, public CountableModel
{
   public:
      ChannelHayKv31CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
