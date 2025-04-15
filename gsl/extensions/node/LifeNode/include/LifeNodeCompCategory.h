// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef LifeNodeCompCategory_H
#define LifeNodeCompCategory_H

#include "Lens.h"
#include "CG_LifeNodeCompCategory.h"

class NDPairList;

class LifeNodeCompCategory : public CG_LifeNodeCompCategory
{
   public:
      LifeNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
