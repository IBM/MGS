#ifndef ChannelCaR_GHKCompCategory_H
#define ChannelCaR_GHKCompCategory_H

#include "Lens.h"
#include "CG_ChannelCaR_GHKCompCategory.h"

#include "CountableModel.h"  // new

class NDPairList;

class ChannelCaR_GHKCompCategory : public CG_ChannelCaR_GHKCompCategory,
                                   public CountableModel
{
  public:
  ChannelCaR_GHKCompCategory(Simulation& sim, const std::string& modelName,
                             const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
