#ifndef ChannelNat_AISCompCategory_H
#define ChannelNat_AISCompCategory_H

#include "Lens.h"
#include "CG_ChannelNat_AISCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelNat_AISCompCategory : public CG_ChannelNat_AISCompCategory,
                               public CountableModel
{
   public:
      ChannelNat_AISCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
