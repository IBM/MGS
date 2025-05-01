// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef HodgkinHuxleyVoltageCompCategory_H
#define HodgkinHuxleyVoltageCompCategory_H

#include "Mgs.h"
#include "CG_HodgkinHuxleyVoltageCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class HodgkinHuxleyVoltageCompCategory
    : public CG_HodgkinHuxleyVoltageCompCategory,
      public CountableModel
{
  public:
  HodgkinHuxleyVoltageCompCategory(Simulation& sim,
                                   const std::string& modelName,
                                   const NDPairList& ndpList);
  void count();
};

#endif
