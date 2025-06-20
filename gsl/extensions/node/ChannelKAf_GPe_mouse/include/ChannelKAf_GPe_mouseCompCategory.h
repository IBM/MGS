// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ChannelKAf_GPe_mouseCompCategory_H
#define ChannelKAf_GPe_mouseCompCategory_H

#include "Mgs.h"
#include "CG_ChannelKAf_GPe_mouseCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKAf_GPe_mouseCompCategory : public CG_ChannelKAf_GPe_mouseCompCategory,
					public CountableModel
{
   public:
      ChannelKAf_GPe_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
