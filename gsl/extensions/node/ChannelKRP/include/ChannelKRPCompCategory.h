#ifndef ChannelKRPCompCategory_H
#define ChannelKRPCompCategory_H

#include "Lens.h"
#include "CG_ChannelKRPCompCategory.h"

#include "CountableModel.h" //new

class NDPairList;

class ChannelKRPCompCategory : public CG_ChannelKRPCompCategory,
                               public CountableModel

{
  public:
  ChannelKRPCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
