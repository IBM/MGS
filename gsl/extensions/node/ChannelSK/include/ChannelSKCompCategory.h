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


#ifndef ChannelSKCompCategory_H
#define ChannelSKCompCategory_H

#include "Lens.h"
#include "CG_ChannelSKCompCategory.h"

#include "CountableModel.h"  //new

class NDPairList;

class ChannelSKCompCategory : public CG_ChannelSKCompCategory,
                              public CountableModel
{
  public:
  ChannelSKCompCategory(Simulation& sim, const std::string& modelName,
                        const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
