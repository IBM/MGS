#ifndef ChannelKCNK_GHKCompCategory_H
#define ChannelKCNK_GHKCompCategory_H

#include "CG_ChannelKCNK_GHKCompCategory.h"
#include "CountableModel.h"
#include "Lens.h"

class NDPairList;

class ChannelKCNK_GHKCompCategory : public CG_ChannelKCNK_GHKCompCategory,
                                    public CountableModel
{
  public:
  ChannelKCNK_GHKCompCategory(Simulation& sim, const std::string& modelName,
                              const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
