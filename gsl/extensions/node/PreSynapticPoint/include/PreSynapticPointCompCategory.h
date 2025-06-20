// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PreSynapticPointCompCategory_H
#define PreSynapticPointCompCategory_H

#include "Mgs.h"
#include "CG_PreSynapticPointCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class PreSynapticPointCompCategory : public CG_PreSynapticPointCompCategory,
                                     public CountableModel
{
  public:
  PreSynapticPointCompCategory(Simulation& sim, const std::string& modelName,
                               const NDPairList& ndpList);
  void count();
};

#endif
