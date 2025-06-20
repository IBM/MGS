// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ForwardSolvePoint4CompCategory_H
#define ForwardSolvePoint4CompCategory_H

#include "Mgs.h"
#include "CG_ForwardSolvePoint4CompCategory.h"

class NDPairList;

class ForwardSolvePoint4CompCategory : public CG_ForwardSolvePoint4CompCategory
{
   public:
      ForwardSolvePoint4CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
