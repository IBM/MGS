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

#ifndef CaConcentrationJunctionPointCompCategory_H
#define CaConcentrationJunctionPointCompCategory_H

#include "Lens.h"
#include "CG_CaConcentrationJunctionPointCompCategory.h"

class NDPairList;

class CaConcentrationJunctionPointCompCategory : public CG_CaConcentrationJunctionPointCompCategory
{
   public:
      CaConcentrationJunctionPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
