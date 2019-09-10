// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2018
//
// (C) Copyright IBM Corp. 2005-2018  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef FSIIAFUnitCompCategory_H
#define FSIIAFUnitCompCategory_H

#include "Lens.h"
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
