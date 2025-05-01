// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef VoltageMegaSynapticSpaceCompCategory_H
#define VoltageMegaSynapticSpaceCompCategory_H

#include "Mgs.h"
#include "CG_VoltageMegaSynapticSpaceCompCategory.h"

class NDPairList;

class VoltageMegaSynapticSpaceCompCategory : public CG_VoltageMegaSynapticSpaceCompCategory
{
   public:
      VoltageMegaSynapticSpaceCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
