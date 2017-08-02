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

#ifndef PreSynapticPointCompCategory_H
#define PreSynapticPointCompCategory_H

#include "Lens.h"
#include "CG_PreSynapticPointCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class PreSynapticPointCompCategory : public CG_PreSynapticPointCompCategory,
                                     public CountableModel
{
  public:
  PreSynapticPointCompCategory(Simulation& sim, const std::string& modelName,
                               const NDPairList& ndpList);
  void count();
};

#endif
