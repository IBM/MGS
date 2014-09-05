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

#ifndef PreSynapticPointCompCategory_H
#define PreSynapticPointCompCategory_H

#include "Lens.h"
#include "CG_PreSynapticPointCompCategory.h"

class NDPairList;

class PreSynapticPointCompCategory : public CG_PreSynapticPointCompCategory
{
   public:
      PreSynapticPointCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
