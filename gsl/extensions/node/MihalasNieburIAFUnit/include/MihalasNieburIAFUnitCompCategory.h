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

#ifndef MihalasNieburIAFUnitCompCategory_H
#define MihalasNieburIAFUnitCompCategory_H

#include "Lens.h"
#include "CG_MihalasNieburIAFUnitCompCategory.h"

class NDPairList;

class MihalasNieburIAFUnitCompCategory : public CG_MihalasNieburIAFUnitCompCategory
{
   public:
      MihalasNieburIAFUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void outputWeightsShared(RNG& rng);
};

#endif
