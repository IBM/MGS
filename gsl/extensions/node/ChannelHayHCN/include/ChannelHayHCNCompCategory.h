// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelHayHCNCompCategory_H
#define ChannelHayHCNCompCategory_H

#include "Lens.h"
#include "CG_ChannelHayHCNCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelHayHCNCompCategory : public CG_ChannelHayHCNCompCategory, public CountableModel
{
   public:
      ChannelHayHCNCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void count();      
};

#endif
