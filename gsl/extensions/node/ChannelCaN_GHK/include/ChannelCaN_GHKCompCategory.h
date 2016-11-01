#ifndef ChannelCaN_GHKCompCategory_H
#define ChannelCaN_GHKCompCategory_H

#include "Lens.h"
#include "CG_ChannelCaN_GHKCompCategory.h"

#include "CountableModel.h"  // new

class NDPairList;

class ChannelCaN_GHKCompCategory : public CG_ChannelCaN_GHKCompCategory,
                                   public CountableModel
{
  public:
  ChannelCaN_GHKCompCategory(Simulation& sim, const std::string& modelName,
                             const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
