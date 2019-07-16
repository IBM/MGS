// =================================================================
//
// (C) Copyright 2018 New Jersey Institute of Technology.
//
// =================================================================


#ifndef ChannelNat_STR_FSI_mouseCompCategory_H
#define ChannelNat_STR_FSI_mouseCompCategory_H

#include "Lens.h"
#include "CG_ChannelNat_STR_FSI_mouseCompCategory.h"
#include "CountableModel.h"
class NDPairList;

class ChannelNat_STR_FSI_mouseCompCategory : public CG_ChannelNat_STR_FSI_mouseCompCategory,
					     public CountableModel
{
   public:
      ChannelNat_STR_FSI_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
