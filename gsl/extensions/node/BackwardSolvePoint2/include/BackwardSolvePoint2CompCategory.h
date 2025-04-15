// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BackwardSolvePoint2CompCategory_H
#define BackwardSolvePoint2CompCategory_H

#include "Lens.h"
#include "CG_BackwardSolvePoint2CompCategory.h"

class NDPairList;

class BackwardSolvePoint2CompCategory : public CG_BackwardSolvePoint2CompCategory
{
   public:
      BackwardSolvePoint2CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
