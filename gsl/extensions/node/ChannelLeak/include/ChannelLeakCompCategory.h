// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
// 
// =============================================================================

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
