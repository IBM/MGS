// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef CaERConcentrationJunctionPointCompCategory_H
#define CaERConcentrationJunctionPointCompCategory_H

#include "Mgs.h"
#include "CG_CaERConcentrationJunctionPointCompCategory.h"

class NDPairList;

class CaERConcentrationJunctionPointCompCategory : public CG_CaERConcentrationJunctionPointCompCategory
{
   public:
      CaERConcentrationJunctionPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
