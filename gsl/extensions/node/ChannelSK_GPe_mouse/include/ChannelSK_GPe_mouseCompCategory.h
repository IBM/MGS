// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#ifndef ChannelSK_GPe_mouseCompCategory_H
#define ChannelSK_GPe_mouseCompCategory_H

#include "Lens.h"
#include "CG_ChannelSK_GPe_mouseCompCategory.h"

#include "CountableModel.h"  //new
class NDPairList;

class ChannelSK_GPe_mouseCompCategory : public CG_ChannelSK_GPe_mouseCompCategory,
                              public CountableModel
{
   public:
      ChannelSK_GPe_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
