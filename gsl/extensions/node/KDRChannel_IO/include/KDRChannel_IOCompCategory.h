// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef KDRChannel_IOCompCategory_H
#define KDRChannel_IOCompCategory_H

#include "Lens.h"
#include "CG_KDRChannel_IOCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class KDRChannel_IOCompCategory : public CG_KDRChannel_IOCompCategory, public CountableModel
{
   public:
      KDRChannel_IOCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE_KDR(RNG& rng);
      void count();      
};

#endif
