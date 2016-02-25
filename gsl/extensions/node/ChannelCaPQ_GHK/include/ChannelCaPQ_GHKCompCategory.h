#ifndef ChannelCaPQ_GHKCompCategory_H
#define ChannelCaPQ_GHKCompCategory_H

#include "Lens.h"
#include "CG_ChannelCaPQ_GHKCompCategory.h"

#include "CountableModel.h"  // new

class NDPairList;

class ChannelCaPQ_GHKCompCategory : public CG_ChannelCaPQ_GHKCompCategory,
                                    public CountableModel
{
  public:
  ChannelCaPQ_GHKCompCategory(Simulation& sim, const std::string& modelName,
                              const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
