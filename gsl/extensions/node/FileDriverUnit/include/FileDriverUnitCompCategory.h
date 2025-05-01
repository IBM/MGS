// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef FileDriverUnitCompCategory_H
#define FileDriverUnitCompCategory_H

#include "Mgs.h"
#include "CG_FileDriverUnitCompCategory.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

class NDPairList;

class FileDriverUnitCompCategory : public CG_FileDriverUnitCompCategory
{
 public:
  FileDriverUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
  void initializeShared(RNG& rng);
  void readInputFile(RNG& rng);
 private:
  std::ifstream* ifs;
  std::string* line;
};

#endif
