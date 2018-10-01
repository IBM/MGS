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


#ifndef ChannelKv31CompCategory_H
#define ChannelKv31CompCategory_H

#include "Lens.h"
#include "CG_ChannelKv31CompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKv31CompCategory : public CG_ChannelKv31CompCategory,
                                public CountableModel
{
  public:
  ChannelKv31CompCategory(Simulation& sim, const std::string& modelName,
                          const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
