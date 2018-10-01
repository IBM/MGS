// =================================================================
//
// (C) Copyright 2018 New Jersey Institute of Technology.
//
// =================================================================


#ifndef ChannelKDR_GPe_mouseCompCategory_H
#define ChannelKDR_GPe_mouseCompCategory_H

#include "Lens.h"
#include "CG_ChannelKDR_GPe_mouseCompCategory.h"
#include "CountableModel.h"
class NDPairList;

class ChannelKDR_GPe_mouseCompCategory : public CG_ChannelKDR_GPe_mouseCompCategory,
					 public CountableModel
{
   public:
      ChannelKDR_GPe_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
