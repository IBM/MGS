// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef ChannelCaLVACompCategory_H
#define ChannelCaLVACompCategory_H

#include "Lens.h"
#include "CG_ChannelCaLVACompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelCaLVACompCategory : public CG_ChannelCaLVACompCategory,
                                 public CountableModel
{
  public:
  ChannelCaLVACompCategory(Simulation& sim, const std::string& modelName,
                           const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
