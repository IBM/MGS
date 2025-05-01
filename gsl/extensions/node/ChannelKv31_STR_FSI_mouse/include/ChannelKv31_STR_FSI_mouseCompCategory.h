// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ChannelKv31_STR_FSI_mouseCompCategory_H
#define ChannelKv31_STR_FSI_mouseCompCategory_H

#include "Mgs.h"
#include "CG_ChannelKv31_STR_FSI_mouseCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKv31_STR_FSI_mouseCompCategory : public CG_ChannelKv31_STR_FSI_mouseCompCategory,

					  public CountableModel
{
   public:
      ChannelKv31_STR_FSI_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
