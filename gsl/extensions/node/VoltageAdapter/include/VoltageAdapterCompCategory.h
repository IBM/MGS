// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef VoltageAdapterCompCategory_H
#define VoltageAdapterCompCategory_H

#include "Mgs.h"
#include "CG_VoltageAdapterCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class VoltageAdapterCompCategory : public CG_VoltageAdapterCompCategory,
   public CountableModel
{
   public:
      VoltageAdapterCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void count();
};

#endif
