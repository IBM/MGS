// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef TraubIAFUnitCompCategory_H
#define TraubIAFUnitCompCategory_H

#include "Lens.h"
#include "CG_TraubIAFUnitCompCategory.h"

class NDPairList;

class TraubIAFUnitCompCategory : public CG_TraubIAFUnitCompCategory
{
   public:
      TraubIAFUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
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
