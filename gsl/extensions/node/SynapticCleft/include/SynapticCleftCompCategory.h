#ifndef SynapticCleftCompCategory_H
#define SynapticCleftCompCategory_H
// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2018  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "CG_SynapticCleftCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class SynapticCleftCompCategory : public CG_SynapticCleftCompCategory,
                                  public CountableModel
{
  public:
  SynapticCleftCompCategory(Simulation& sim, const std::string& modelName,
                            const NDPairList& ndpList);
  void count();
};

#endif
