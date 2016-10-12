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

#ifndef IP3ConcentrationCompCategory_H
#define IP3ConcentrationCompCategory_H

#include "Lens.h"
#include "CG_IP3ConcentrationCompCategory.h"
#include "CountableModel.h"

#include "NTSMacros.h"

class NDPairList;

class IP3ConcentrationCompCategory : public CG_IP3ConcentrationCompCategory,
                                    public CountableModel
{
  public:
  IP3ConcentrationCompCategory(Simulation& sim, const std::string& modelName,
                              const NDPairList& ndpList);
  void deriveParameters(RNG& rng);
  void count();
};

#endif
