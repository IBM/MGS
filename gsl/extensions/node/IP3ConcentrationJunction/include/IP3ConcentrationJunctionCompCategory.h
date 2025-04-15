// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef IP3ConcentrationJunctionCompCategory_H
#define IP3ConcentrationJunctionCompCategory_H

#include "Lens.h"
#include "CG_IP3ConcentrationJunctionCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class IP3ConcentrationJunctionCompCategory : public CG_IP3ConcentrationJunctionCompCategory, public CountableModel
{
   public:
      IP3ConcentrationJunctionCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void deriveParameters(RNG& rng);
      void count();
};

#endif
