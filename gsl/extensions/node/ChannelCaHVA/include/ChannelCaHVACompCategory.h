#ifndef ChannelCaHVACompCategory_H
#define ChannelCaHVACompCategory_H

#include "Lens.h"
#include "CG_ChannelCaHVACompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelCaHVACompCategory : public CG_ChannelCaHVACompCategory,
                               public CountableModel
{
   public:
      ChannelCaHVACompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeTadj(RNG& rng);
      void count();
};

#endif
