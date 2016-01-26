#ifndef ChannelNatCompCategory_H
#define ChannelNatCompCategory_H

#include "Lens.h"
#include "CG_ChannelNatCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelNatCompCategory : public CG_ChannelNatCompCategory,
                               public CountableModel
{
  public:
  ChannelNatCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
