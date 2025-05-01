// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BackwardSolvePoint6CompCategory_H
#define BackwardSolvePoint6CompCategory_H

#include "Mgs.h"
#include "CG_BackwardSolvePoint6CompCategory.h"

class NDPairList;

class BackwardSolvePoint6CompCategory : public CG_BackwardSolvePoint6CompCategory
{
   public:
      BackwardSolvePoint6CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
