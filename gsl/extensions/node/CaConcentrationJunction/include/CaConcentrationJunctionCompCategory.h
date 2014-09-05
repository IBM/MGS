// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef CaConcentrationJunctionCompCategory_H
#define CaConcentrationJunctionCompCategory_H

#include "Lens.h"
#include "CG_CaConcentrationJunctionCompCategory.h"
#include "../../../../../nti/CountableModel.h"

class NDPairList;

class CaConcentrationJunctionCompCategory : public CG_CaConcentrationJunctionCompCategory, public CountableModel
{
   public:
      CaConcentrationJunctionCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void deriveParameters(RNG& rng);
      void count();
};

#endif
