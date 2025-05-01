// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef DNEdgeSetCompCategory_H
#define DNEdgeSetCompCategory_H

#include "Mgs.h"
#include "CG_DNEdgeSetCompCategory.h"
#include "TransferFunction.h"

class NDPairList;

class DNEdgeSetCompCategory : public CG_DNEdgeSetCompCategory
{
   public:
      DNEdgeSetCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
