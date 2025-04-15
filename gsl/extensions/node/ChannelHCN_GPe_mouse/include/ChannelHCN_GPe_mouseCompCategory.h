// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#ifndef ChannelHCN_GPe_mouseCompCategory_H
#define ChannelHCN_GPe_mouseCompCategory_H

#include "Lens.h"
#include "CG_ChannelHCN_GPe_mouseCompCategory.h"


#include "CountableModel.h"
class NDPairList;

class ChannelHCN_GPe_mouseCompCategory : public CG_ChannelHCN_GPe_mouseCompCategory,
                               public CountableModel
{
   public:
      ChannelHCN_GPe_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeTadj(RNG& rng);
      void count();
};

#endif
