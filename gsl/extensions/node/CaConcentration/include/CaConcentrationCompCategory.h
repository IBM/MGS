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

#ifndef CaConcentrationCompCategory_H
#define CaConcentrationCompCategory_H

#include "Lens.h"
#include "CG_CaConcentrationCompCategory.h"
#include "CountableModel.h"

#include "NTSMacros.h"

class NDPairList;

class CaConcentrationCompCategory : public CG_CaConcentrationCompCategory,
                                    public CountableModel
{
  public:
  CaConcentrationCompCategory(Simulation& sim, const std::string& modelName,
                              const NDPairList& ndpList);
  void deriveParameters(RNG& rng);
  void count();
};

#endif
