// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SpineIAFUnitCompCategory_H
#define SpineIAFUnitCompCategory_H

#include "Lens.h"
#include "CG_SpineIAFUnitCompCategory.h"

class NDPairList;

class SpineIAFUnitCompCategory : public CG_SpineIAFUnitCompCategory
{
 public:
  SpineIAFUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
  void initializeShared(RNG& rng);
  void outputWeightsShared(RNG& rng);
 private:
  std::ofstream* weights_file;
  std::ostringstream os_weights;
};

#endif
