// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef FSIIAFUnitCompCategory_H
#define FSIIAFUnitCompCategory_H

#include "Mgs.h"
#include "CG_FSIIAFUnitCompCategory.h"

class NDPairList;

class FSIIAFUnitCompCategory : public CG_FSIIAFUnitCompCategory
{
   public:
      FSIIAFUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void outputPSPsShared(RNG& rng);
 private:
      std::ofstream* weight_file;      
      std::ofstream* GJ_file;      
      std::ofstream* psp_file;
      std::ostringstream os_weight;        
      std::ostringstream os_GJ;        
      std::ostringstream os_psp;
};

#endif
