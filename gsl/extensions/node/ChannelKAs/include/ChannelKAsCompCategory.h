/*
=================================================================
Licensed Materials - Property of IBM

"Restricted Materials of IBM"

BMC-YKT-03-25-2018

(C) Copyright IBM Corp. 2005-2017  All rights reserved

US Government Users Restricted Rights -
Use, duplication or disclosure restricted by
GSA ADP Schedule Contract with IBM Corp.

================================================================
*/


#ifndef ChannelKAsCompCategory_H
#define ChannelKAsCompCategory_H

#include "Lens.h"
#include "CG_ChannelKAsCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKAsCompCategory : public CG_ChannelKAsCompCategory,
                               public CountableModel
{
  public:
  ChannelKAsCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
