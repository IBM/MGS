// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#ifndef ChannelKv1_STR_FSI_mouseCompCategory_H
#define ChannelKv1_STR_FSI_mouseCompCategory_H

#include "Mgs.h"
#include "CG_ChannelKv1_STR_FSI_mouseCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKv1_STR_FSI_mouseCompCategory : public CG_ChannelKv1_STR_FSI_mouseCompCategory,
					     public CountableModel
{
   public:
      ChannelKv1_STR_FSI_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
