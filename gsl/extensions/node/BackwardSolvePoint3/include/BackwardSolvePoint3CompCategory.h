// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BackwardSolvePoint3CompCategory_H
#define BackwardSolvePoint3CompCategory_H

#include "Lens.h"
#include "CG_BackwardSolvePoint3CompCategory.h"

class NDPairList;

class BackwardSolvePoint3CompCategory : public CG_BackwardSolvePoint3CompCategory
{
   public:
      BackwardSolvePoint3CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
