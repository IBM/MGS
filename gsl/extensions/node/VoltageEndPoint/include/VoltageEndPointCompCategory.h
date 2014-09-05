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
