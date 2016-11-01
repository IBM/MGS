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

#ifndef CaERConcentrationCompCategory_H
#define CaERConcentrationCompCategory_H

#include "Lens.h"
#include "CG_CaERConcentrationCompCategory.h"
#include "CountableModel.h"

#include "NTSMacros.h"

class NDPairList;

class CaERConcentrationCompCategory : public CG_CaERConcentrationCompCategory,
                                      public CountableModel
{
  public:
  CaERConcentrationCompCategory(Simulation& sim, const std::string& modelName,
                                const NDPairList& ndpList);
  void deriveParameters(RNG& rng);
  void count();
};

#endif
