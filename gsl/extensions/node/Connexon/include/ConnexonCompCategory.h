// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ConnexonCompCategory_H
#define ConnexonCompCategory_H

#include "Mgs.h"
#include "CG_ConnexonCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ConnexonCompCategory : public CG_ConnexonCompCategory,
                             public CountableModel
{
  public:
  ConnexonCompCategory(Simulation& sim, const std::string& modelName,
                       const NDPairList& ndpList);
  void count();
};

#endif
