#ifndef SynapticCleftCompCategory_H
#define SynapticCleftCompCategory_H
// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "CG_SynapticCleftCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class SynapticCleftCompCategory : public CG_SynapticCleftCompCategory,
                                  public CountableModel
{
  public:
  SynapticCleftCompCategory(Simulation& sim, const std::string& modelName,
                            const NDPairList& ndpList);
  void count();
};

#endif
