#ifndef ChannelCaT_GHKCompCategory_H
#define ChannelCaT_GHKCompCategory_H

#include "Lens.h"
#include "CG_ChannelCaT_GHKCompCategory.h"

#include "CountableModel.h"  // new

class NDPairList;

class ChannelCaT_GHKCompCategory : public CG_ChannelCaT_GHKCompCategory,
                               public CountableModel
{
  public:
  ChannelCaT_GHKCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
