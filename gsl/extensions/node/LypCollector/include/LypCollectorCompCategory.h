// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef LypCollectorCompCategory_H
#define LypCollectorCompCategory_H

#include "Mgs.h"
#include "CG_LypCollectorCompCategory.h"

class NDPairList;

class LypCollectorCompCategory : public CG_LypCollectorCompCategory
{
   public:
      LypCollectorCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
