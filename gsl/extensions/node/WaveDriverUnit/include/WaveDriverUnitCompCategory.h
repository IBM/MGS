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

#ifndef WaveDriverUnitCompCategory_H
#define WaveDriverUnitCompCategory_H

#include "Lens.h"
#include "CG_WaveDriverUnitCompCategory.h"

class NDPairList;

class WaveDriverUnitCompCategory : public CG_WaveDriverUnitCompCategory
{
 public:
  WaveDriverUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
  void initializeShared(RNG& rng);
};

#endif
