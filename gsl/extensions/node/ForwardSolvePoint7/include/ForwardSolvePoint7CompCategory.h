// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ForwardSolvePoint7CompCategory_H
#define ForwardSolvePoint7CompCategory_H

#include "Lens.h"
#include "CG_ForwardSolvePoint7CompCategory.h"

class NDPairList;

class ForwardSolvePoint7CompCategory : public CG_ForwardSolvePoint7CompCategory
{
   public:
      ForwardSolvePoint7CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
