// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef IzhikUnitCompCategory_H
#define IzhikUnitCompCategory_H

#include "Mgs.h"
#include "CG_IzhikUnitCompCategory.h"

class NDPairList;

class IzhikUnitCompCategory : public CG_IzhikUnitCompCategory
{
   public:
      IzhikUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
