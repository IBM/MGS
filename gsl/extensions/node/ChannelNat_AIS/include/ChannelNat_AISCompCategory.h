#ifndef ChannelNat_AISCompCategory_H
#define ChannelNat_AISCompCategory_H

#include "Lens.h"
#include "CG_ChannelNat_AISCompCategory.h"

class NDPairList;

class ChannelNat_AISCompCategory : public CG_ChannelNat_AISCompCategory
{
   public:
      ChannelNat_AISCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
};

#endif
