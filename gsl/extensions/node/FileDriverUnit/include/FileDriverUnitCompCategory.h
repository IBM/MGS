// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef FileDriverUnitCompCategory_H
#define FileDriverUnitCompCategory_H

#include "Lens.h"
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
