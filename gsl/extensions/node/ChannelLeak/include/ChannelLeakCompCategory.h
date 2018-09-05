#ifndef ChannelLeakCompCategory_H
#define ChannelLeakCompCategory_H

#include "Lens.h"
#include "CG_ChannelLeakCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelLeakCompCategory : public CG_ChannelLeakCompCategory,
				public CountableModel
{
   public:
      ChannelLeakCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
	void count();      
};

#endif
