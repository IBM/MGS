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


#ifndef ChannelKAfCompCategory_H
#define ChannelKAfCompCategory_H

#include "Lens.h"
#include "CG_ChannelKAfCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKAfCompCategory : public CG_ChannelKAfCompCategory,
                               public CountableModel
{
  public:
  ChannelKAfCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
