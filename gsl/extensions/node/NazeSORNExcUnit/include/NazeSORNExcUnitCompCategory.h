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

#ifndef NazeSORNExcUnitCompCategory_H
#define NazeSORNExcUnitCompCategory_H

#include "Lens.h"
#include "CG_NazeSORNExcUnitCompCategory.h"

class NDPairList;

class NazeSORNExcUnitCompCategory : public CG_NazeSORNExcUnitCompCategory
{
   public:
      NazeSORNExcUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void saveInitParams(RNG& rng);
      void outputWeightsShared(RNG& rng);
      void outputDelaysShared(RNG& rng);
      void inputWeightsShared(RNG& rng);
};

#endif
