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

#ifndef LinskerInfomaxUnitCompCategory_H
#define LinskerInfomaxUnitCompCategory_H

#include "Lens.h"
#include "CG_LinskerInfomaxUnitCompCategory.h"

class NDPairList;

class LinskerInfomaxUnitCompCategory : public CG_LinskerInfomaxUnitCompCategory
{
   public:
      LinskerInfomaxUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void outputWeightsShared(RNG& rng);
      void invertQmatrixShared(RNG& rng);
};

#endif
