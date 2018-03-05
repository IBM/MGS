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

#ifndef NazeSORNInhUnitCompCategory_H
#define NazeSORNInhUnitCompCategory_H

#include "Lens.h"
#include "CG_NazeSORNInhUnitCompCategory.h"

class NDPairList;

class NazeSORNInhUnitCompCategory : public CG_NazeSORNInhUnitCompCategory
{
   public:
      NazeSORNInhUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void inputWeightsShared(RNG& rng);
      void outputWeightsShared(RNG& rng); 
};

#endif
