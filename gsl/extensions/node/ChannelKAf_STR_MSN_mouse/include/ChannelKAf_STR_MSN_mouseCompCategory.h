// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================
#ifndef ChannelKAf_STR_MSN_mouseCompCategory_H
#define ChannelKAf_STR_MSN_mouseCompCategory_H

#include "Mgs.h"
#include "CG_ChannelKAf_STR_MSN_mouseCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKAf_STR_MSN_mouseCompCategory : public CG_ChannelKAf_STR_MSN_mouseCompCategory, public CountableModel
{
   public:
      ChannelKAf_STR_MSN_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
