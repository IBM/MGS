#ifndef ChannelBKalphabetaCompCategory_H
#define ChannelBKalphabetaCompCategory_H

#include "Lens.h"
#include "CG_ChannelBKalphabetaCompCategory.h"

#include "CountableModel.h" //new

class NDPairList;

class ChannelBKalphabetaCompCategory : public CG_ChannelBKalphabetaCompCategory,
                                   public CountableModel
{
  public:
  ChannelBKalphabetaCompCategory(Simulation& sim, const std::string& modelName,
                             const NDPairList& ndpList);
  void computeE(RNG& rng);
	void count();
};

#endif
