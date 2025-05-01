// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef KCaChannelCompCategory_H
#define KCaChannelCompCategory_H

#include "Mgs.h"
#include "CG_KCaChannelCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class KCaChannelCompCategory : public CG_KCaChannelCompCategory, public CountableModel
{
   public:
      KCaChannelCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE_K(RNG& rng);
      void count();      
};

#endif
