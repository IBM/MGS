// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef MotoneuronUnitCompCategory_H
#define MotoneuronUnitCompCategory_H

#include "Lens.h"
#include "CG_MotoneuronUnitCompCategory.h"

class NDPairList;

class MotoneuronUnitCompCategory : public CG_MotoneuronUnitCompCategory
{
 public:
  MotoneuronUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
  void initializeShared(RNG& rng);
};

#endif
