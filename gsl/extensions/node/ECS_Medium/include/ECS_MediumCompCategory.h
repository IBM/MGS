// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ECS_MediumCompCategory_H
#define ECS_MediumCompCategory_H

#include "Mgs.h"
#include "CG_ECS_MediumCompCategory.h"

class NDPairList;

class ECS_MediumCompCategory : public CG_ECS_MediumCompCategory
{
   public:
      ECS_MediumCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
