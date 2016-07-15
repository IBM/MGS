#ifndef ChannelKDR_AISCompCategory_H
#define ChannelKDR_AISCompCategory_H

#include "Lens.h"
#include "CG_ChannelKDR_AISCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKDR_AISCompCategory : public CG_ChannelKDR_AISCompCategory,
                                   public CountableModel
{
   public:
      ChannelKDR_AISCompCategory(Simulation& sim, const std::string& modelName, 
                                 const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
