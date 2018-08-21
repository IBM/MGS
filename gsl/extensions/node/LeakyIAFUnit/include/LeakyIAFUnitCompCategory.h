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

#ifndef LeakyIAFUnitCompCategory_H
#define LeakyIAFUnitCompCategory_H

#include "Lens.h"
#include "CG_LeakyIAFUnitCompCategory.h"

class NDPairList;

class LeakyIAFUnitCompCategory : public CG_LeakyIAFUnitCompCategory
{
 public:
  LeakyIAFUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
  void initializeShared(RNG& rng);
};

#endif
