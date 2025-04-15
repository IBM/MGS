// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef WaveDriverUnitCompCategory_H
#define WaveDriverUnitCompCategory_H

#include "Lens.h"
#include "CG_WaveDriverUnitCompCategory.h"

class NDPairList;

class WaveDriverUnitCompCategory : public CG_WaveDriverUnitCompCategory
{
 public:
  WaveDriverUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
