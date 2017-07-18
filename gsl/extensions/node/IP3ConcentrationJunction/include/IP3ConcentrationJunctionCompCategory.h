// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef IP3ConcentrationJunctionCompCategory_H
#define IP3ConcentrationJunctionCompCategory_H

#include "Lens.h"
#include "CG_IP3ConcentrationJunctionCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class IP3ConcentrationJunctionCompCategory : public CG_IP3ConcentrationJunctionCompCategory, public CountableModel
{
   public:
      IP3ConcentrationJunctionCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void deriveParameters(RNG& rng);
      void count();
};

#endif
