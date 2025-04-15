// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ForwardSolvePoint2CompCategory_H
#define ForwardSolvePoint2CompCategory_H

#include "Lens.h"
#include "CG_ForwardSolvePoint2CompCategory.h"

class NDPairList;

class ForwardSolvePoint2CompCategory : public CG_ForwardSolvePoint2CompCategory
{
   public:
      ForwardSolvePoint2CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
