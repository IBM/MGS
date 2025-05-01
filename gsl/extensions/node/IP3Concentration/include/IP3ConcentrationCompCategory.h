// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef IP3ConcentrationCompCategory_H
#define IP3ConcentrationCompCategory_H

#include "Mgs.h"
#include "CG_IP3ConcentrationCompCategory.h"
#include "CountableModel.h"

#include "NTSMacros.h"

class NDPairList;

class IP3ConcentrationCompCategory : public CG_IP3ConcentrationCompCategory,
                                    public CountableModel
{
  public:
  IP3ConcentrationCompCategory(Simulation& sim, const std::string& modelName,
                              const NDPairList& ndpList);
  void deriveParameters(RNG& rng);
  void count();
};

#endif
