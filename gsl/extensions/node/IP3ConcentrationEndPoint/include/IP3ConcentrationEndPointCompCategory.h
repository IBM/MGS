// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef IP3ConcentrationEndPointCompCategory_H
#define IP3ConcentrationEndPointCompCategory_H

#include "Mgs.h"
#include "CG_IP3ConcentrationEndPointCompCategory.h"

class NDPairList;

class IP3ConcentrationEndPointCompCategory : public CG_IP3ConcentrationEndPointCompCategory
{
   public:
      IP3ConcentrationEndPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
