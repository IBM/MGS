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

#ifndef GABAAReceptorCompCategory_H
#define GABAAReceptorCompCategory_H

#include "Lens.h"
#include "CG_GABAAReceptorCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class GABAAReceptorCompCategory : public CG_GABAAReceptorCompCategory,
                                  public CountableModel
{
  public:
  GABAAReceptorCompCategory(Simulation& sim, const std::string& modelName,
                            const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
