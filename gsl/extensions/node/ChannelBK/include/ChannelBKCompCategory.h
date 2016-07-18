#ifndef ChannelBKCompCategory_H
#define ChannelBKCompCategory_H

#include "Lens.h"
#include "CG_ChannelBKCompCategory.h"

#include "CountableModel.h"

class NDPairList;

class ChannelBKCompCategory : public CG_ChannelBKCompCategory, public CountableModel
{
   public:
      ChannelBKCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
