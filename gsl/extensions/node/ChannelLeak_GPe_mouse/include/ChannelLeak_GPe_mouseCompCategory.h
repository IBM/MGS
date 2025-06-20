// =============================================================================
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
//
// =============================================================================

#ifndef ChannelLeak_GPe_mouseCompCategory_H
#define ChannelLeak_GPe_mouseCompCategory_H

#include "Mgs.h"
#include "CG_ChannelLeak_GPe_mouseCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelLeak_GPe_mouseCompCategory : public CG_ChannelLeak_GPe_mouseCompCategory,
					  public CountableModel
{
   public:
      ChannelLeak_GPe_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void count();
};

#endif
