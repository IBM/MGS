#ifndef ChannelNapCompCategory_H
#define ChannelNapCompCategory_H

#include "Lens.h"
#include "CG_ChannelNapCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelNapCompCategory : public CG_ChannelNapCompCategory,
                               public CountableModel
{
  public:
  ChannelNapCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
