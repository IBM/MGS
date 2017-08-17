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

#ifndef ConnexonCompCategory_H
#define ConnexonCompCategory_H

#include "Lens.h"
#include "CG_ConnexonCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ConnexonCompCategory : public CG_ConnexonCompCategory,
                             public CountableModel
{
  public:
  ConnexonCompCategory(Simulation& sim, const std::string& modelName,
                       const NDPairList& ndpList);
  void count();
};

#endif
