#ifndef ChannelKAfCompCategory_H
#define ChannelKAfCompCategory_H

#include "Lens.h"
#include "CG_ChannelKAfCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKAfCompCategory : public CG_ChannelKAfCompCategory,
                               public CountableModel
{
  public:
  ChannelKAfCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
