#ifndef ChannelKv31_STR_FSI_mouseCompCategory_H
#define ChannelKv31_STR_FSI_mouseCompCategory_H

#include "Lens.h"
#include "CG_ChannelKv31_STR_FSI_mouseCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKv31_STR_FSI_mouseCompCategory : public CG_ChannelKv31_STR_FSI_mouseCompCategory,

					  public CountableModel
{
   public:
      ChannelKv31_STR_FSI_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
