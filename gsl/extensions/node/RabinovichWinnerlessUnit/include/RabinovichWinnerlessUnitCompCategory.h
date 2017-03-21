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

#ifndef RabinovichWinnerlessUnitCompCategory_H
#define RabinovichWinnerlessUnitCompCategory_H

#include "Lens.h"
#include "CG_RabinovichWinnerlessUnitCompCategory.h"

class NDPairList;

class RabinovichWinnerlessUnitCompCategory : public CG_RabinovichWinnerlessUnitCompCategory
{
   public:
      RabinovichWinnerlessUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void outputWeightsShared(RNG& rng);
};

#endif
