#ifndef ChannelKAf_KChIPCompCategory_H
#define ChannelKAf_KChIPCompCategory_H

#include "Lens.h"
#include "CG_ChannelKAf_KChIPCompCategory.h"

#include "CountableModel.h"

class NDPairList;

class ChannelKAf_KChIPCompCategory : public CG_ChannelKAf_KChIPCompCategory, 
   public CountableModel
{
   public:
      ChannelKAf_KChIPCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
