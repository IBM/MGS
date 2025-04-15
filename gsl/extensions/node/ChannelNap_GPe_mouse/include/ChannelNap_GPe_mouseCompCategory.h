// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#ifndef ChannelNap_GPe_mouseCompCategory_H
#define ChannelNap_GPe_mouseCompCategory_H

#include "Lens.h"
#include "CG_ChannelNap_GPe_mouseCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelNap_GPe_mouseCompCategory : public CG_ChannelNap_GPe_mouseCompCategory,
					 public CountableModel
{
   public:
      ChannelNap_GPe_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
