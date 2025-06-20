// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHayCaHVACompCategory_H
#define ChannelHayCaHVACompCategory_H

#include "Mgs.h"
#include "CG_ChannelHayCaHVACompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelHayCaHVACompCategory : public CG_ChannelHayCaHVACompCategory, public CountableModel
{
   public:
      ChannelHayCaHVACompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void count();   
};

#endif
