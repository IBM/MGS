// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#ifndef ChannelCaHVA_GPe_mouseCompCategory_H
#define ChannelCaHVA_GPe_mouseCompCategory_H

#include "Lens.h"
#include "CG_ChannelCaHVA_GPe_mouseCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelCaHVA_GPe_mouseCompCategory : public CG_ChannelCaHVA_GPe_mouseCompCategory,
					   public CountableModel
{
   public:
      ChannelCaHVA_GPe_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeTadj(RNG& rng);
      void count();
};

#endif
