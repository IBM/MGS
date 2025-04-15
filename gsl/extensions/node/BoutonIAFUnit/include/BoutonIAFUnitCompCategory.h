// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BoutonIAFUnitCompCategory_H
#define BoutonIAFUnitCompCategory_H

#include "Lens.h"
#include "CG_BoutonIAFUnitCompCategory.h"

class NDPairList;

class BoutonIAFUnitCompCategory : public CG_BoutonIAFUnitCompCategory
{
 public:
  BoutonIAFUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
  void initializeShared(RNG& rng);
 private:
  std::ofstream** indexs_file;
  std::ostringstream* os_indexs;
};
#endif
