// =================================================================
//
// (C) Copyright 2018 New Jersey Institute of Technology.
//
// =================================================================


#ifndef ChannelLeak_STR_MSN_mouseCompCategory_H
#define ChannelLeak_STR_MSN_mouseCompCategory_H

#include "Lens.h"
#include "CG_ChannelLeak_STR_MSN_mouseCompCategory.h"

#include "CountableModel.h"
class NDPairList;

class ChannelLeak_STR_MSN_mouseCompCategory : public CG_ChannelLeak_STR_MSN_mouseCompCategory,
					      public CountableModel
{
   public:
      ChannelLeak_STR_MSN_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
	void count();	
};

#endif
