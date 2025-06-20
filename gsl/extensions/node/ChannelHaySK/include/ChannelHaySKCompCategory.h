// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHaySKCompCategory_H
#define ChannelHaySKCompCategory_H

#include "Mgs.h"
#include "CG_ChannelHaySKCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelHaySKCompCategory : public CG_ChannelHaySKCompCategory, public CountableModel
{
   public:
      ChannelHaySKCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();      
};

#endif
