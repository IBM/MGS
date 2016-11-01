#ifndef ChannelKIRCompCategory_H
#define ChannelKIRCompCategory_H

#include "Lens.h"
#include "CG_ChannelKIRCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKIRCompCategory : public CG_ChannelKIRCompCategory,
                               public CountableModel
{
  public:
  ChannelKIRCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
