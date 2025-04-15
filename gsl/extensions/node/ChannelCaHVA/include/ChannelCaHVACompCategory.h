// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ChannelCaHVACompCategory_H
#define ChannelCaHVACompCategory_H

#include "Lens.h"
#include "CG_ChannelCaHVACompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelCaHVACompCategory : public CG_ChannelCaHVACompCategory,
                               public CountableModel
{
   public:
      ChannelCaHVACompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeTadj(RNG& rng);
      void count();
};

#endif
