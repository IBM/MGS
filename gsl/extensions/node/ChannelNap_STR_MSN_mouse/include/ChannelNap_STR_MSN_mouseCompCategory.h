// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#ifndef ChannelNap_STR_MSN_mouseCompCategory_H
#define ChannelNap_STR_MSN_mouseCompCategory_H

#include "Mgs.h"
#include "CG_ChannelNap_STR_MSN_mouseCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelNap_STR_MSN_mouseCompCategory : public CG_ChannelNap_STR_MSN_mouseCompCategory, public CountableModel
{
   public:
      ChannelNap_STR_MSN_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
