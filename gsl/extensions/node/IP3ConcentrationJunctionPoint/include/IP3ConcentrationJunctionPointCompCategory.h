// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef IP3ConcentrationJunctionPointCompCategory_H
#define IP3ConcentrationJunctionPointCompCategory_H

#include "Mgs.h"
#include "CG_IP3ConcentrationJunctionPointCompCategory.h"

class NDPairList;

class IP3ConcentrationJunctionPointCompCategory : public CG_IP3ConcentrationJunctionPointCompCategory
{
   public:
      IP3ConcentrationJunctionPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
