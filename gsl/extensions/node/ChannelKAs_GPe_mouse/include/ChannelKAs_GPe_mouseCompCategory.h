// =================================================================
//
// (C) Copyright 2018 New Jersey Institute of Technology.
//
// =================================================================


#ifndef ChannelKAs_GPe_mouseCompCategory_H
#define ChannelKAs_GPe_mouseCompCategory_H

#include "Lens.h"
#include "CG_ChannelKAs_GPe_mouseCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKAs_GPe_mouseCompCategory : public CG_ChannelKAs_GPe_mouseCompCategory,
					 public CountableModel
{
   public:
      ChannelKAs_GPe_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
