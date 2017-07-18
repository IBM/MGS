// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef HodgkinHuxleyVoltageCompCategory_H
#define HodgkinHuxleyVoltageCompCategory_H

#include "Lens.h"
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
