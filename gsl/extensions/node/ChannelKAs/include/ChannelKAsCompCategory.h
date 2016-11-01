#ifndef ChannelKAsCompCategory_H
#define ChannelKAsCompCategory_H

#include "Lens.h"
#include "CG_ChannelKAsCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKAsCompCategory : public CG_ChannelKAsCompCategory,
                               public CountableModel
{
  public:
  ChannelKAsCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
