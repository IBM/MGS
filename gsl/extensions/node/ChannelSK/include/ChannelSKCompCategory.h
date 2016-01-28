#ifndef ChannelSKCompCategory_H
#define ChannelSKCompCategory_H

#include "Lens.h"
#include "CG_ChannelSKCompCategory.h"

#include "CountableModel.h"  //new

class NDPairList;

class ChannelSKCompCategory : public CG_ChannelSKCompCategory,
                              public CountableModel
{
  public:
  ChannelSKCompCategory(Simulation& sim, const std::string& modelName,
                        const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
