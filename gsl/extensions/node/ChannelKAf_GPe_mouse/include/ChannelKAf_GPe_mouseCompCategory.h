#ifndef ChannelKAf_GPe_mouseCompCategory_H
#define ChannelKAf_GPe_mouseCompCategory_H

#include "Lens.h"
#include "CG_ChannelKAf_GPe_mouseCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKAf_GPe_mouseCompCategory : public CG_ChannelKAf_GPe_mouseCompCategory,
					public CountableModel
{
   public:
      ChannelKAf_GPe_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
