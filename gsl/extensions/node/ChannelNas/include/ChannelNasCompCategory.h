#ifndef ChannelNasCompCategory_H
#define ChannelNasCompCategory_H

#include "CG_ChannelNasCompCategory.h"
#include "CountableModel.h"
#include "Lens.h"

class NDPairList;

class ChannelNasCompCategory : public CG_ChannelNasCompCategory,
                               public CountableModel
{
  public:
  ChannelNasCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
