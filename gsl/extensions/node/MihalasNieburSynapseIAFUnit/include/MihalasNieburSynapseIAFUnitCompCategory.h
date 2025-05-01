// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef MihalasNieburSynapseIAFUnitCompCategory_H
#define MihalasNieburSynapseIAFUnitCompCategory_H

#include "Mgs.h"
#include "CG_MihalasNieburSynapseIAFUnitCompCategory.h"

class NDPairList;

class MihalasNieburSynapseIAFUnitCompCategory : public CG_MihalasNieburSynapseIAFUnitCompCategory
{
 public:
  MihalasNieburSynapseIAFUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
