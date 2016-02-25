#ifndef ChannelCaLv12_GHKCompCategory_H
#define ChannelCaLv12_GHKCompCategory_H

#include "Lens.h"
#include "CG_ChannelCaLv12_GHKCompCategory.h"

#include "CountableModel.h"  // new

class NDPairList;

class ChannelCaLv12_GHKCompCategory : public CG_ChannelCaLv12_GHKCompCategory,
                                      public CountableModel
{
  public:
  ChannelCaLv12_GHKCompCategory(Simulation& sim, const std::string& modelName,
                                const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
