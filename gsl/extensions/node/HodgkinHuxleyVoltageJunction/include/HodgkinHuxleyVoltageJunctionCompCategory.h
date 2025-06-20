// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef HodgkinHuxleyVoltageJunctionCompCategory_H
#define HodgkinHuxleyVoltageJunctionCompCategory_H

#include "Mgs.h"
#include "CG_HodgkinHuxleyVoltageJunctionCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class HodgkinHuxleyVoltageJunctionCompCategory : public CG_HodgkinHuxleyVoltageJunctionCompCategory, public CountableModel
{
   public:
      HodgkinHuxleyVoltageJunctionCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void count();
};

#endif
