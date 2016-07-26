#ifndef ChannelKv31CompCategory_H
#define ChannelKv31CompCategory_H

#include "Lens.h"
#include "CG_ChannelKv31CompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKv31CompCategory : public CG_ChannelKv31CompCategory,
                                public CountableModel
{
  public:
  ChannelKv31CompCategory(Simulation& sim, const std::string& modelName,
                          const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
