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

#ifndef NMDAReceptorCompCategory_H
#define NMDAReceptorCompCategory_H

#include "Lens.h"
#include "CG_NMDAReceptorCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class NMDAReceptorCompCategory : public CG_NMDAReceptorCompCategory,
                                 public CountableModel
{
  public:
  NMDAReceptorCompCategory(Simulation& sim, const std::string& modelName,
                           const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
