// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GoodwinCompCategory_H
#define GoodwinCompCategory_H

#include "Lens.h"
#include "CG_GoodwinCompCategory.h"

class NDPairList;

class GoodwinCompCategory : public CG_GoodwinCompCategory
{
  public:
    GoodwinCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
