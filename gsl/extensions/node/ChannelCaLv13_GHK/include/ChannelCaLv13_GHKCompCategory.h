#ifndef ChannelCaLv13_GHKCompCategory_H
#define ChannelCaLv13_GHKCompCategory_H

#include "Lens.h"
#include "CG_ChannelCaLv13_GHKCompCategory.h"

#include "CountableModel.h"  // new

class NDPairList;

class ChannelCaLv13_GHKCompCategory : public CG_ChannelCaLv13_GHKCompCategory,
                                      public CountableModel
{
  public:
  ChannelCaLv13_GHKCompCategory(Simulation& sim, const std::string& modelName,
                                const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
