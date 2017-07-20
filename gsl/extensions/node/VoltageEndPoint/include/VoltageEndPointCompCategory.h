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

#ifndef VoltageEndPointCompCategory_H
#define VoltageEndPointCompCategory_H

#include "Lens.h"
#include "CG_VoltageEndPointCompCategory.h"

class NDPairList;

class VoltageEndPointCompCategory : public CG_VoltageEndPointCompCategory
{
   public:
      VoltageEndPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
