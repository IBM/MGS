// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef KDRChannel_AISCompCategory_H
#define KDRChannel_AISCompCategory_H

#include "Mgs.h"
#include "CG_KDRChannel_AISCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class KDRChannel_AISCompCategory : public CG_KDRChannel_AISCompCategory, public CountableModel
{
   public:
      KDRChannel_AISCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE_KDR(RNG& rng);
      void count();      
};

#endif
