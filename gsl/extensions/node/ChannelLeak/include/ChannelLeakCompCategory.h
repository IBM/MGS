/* =================================================================
Licensed Materials - Property of IBM

"Restricted Materials of IBM"

BMC-YKT-07-18-2017

(C) Copyright IBM Corp. 2005-2017  All rights reserved

US Government Users Restricted Rights -
Use, duplication or disclosure restricted by
GSA ADP Schedule Contract with IBM Corp.

=================================================================
*/


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
