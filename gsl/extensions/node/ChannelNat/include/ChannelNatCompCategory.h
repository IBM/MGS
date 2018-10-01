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


#ifndef ChannelNatCompCategory_H
#define ChannelNatCompCategory_H

#include "Lens.h"
#include "CG_ChannelNatCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelNatCompCategory : public CG_ChannelNatCompCategory,
                               public CountableModel
{
  public:
  ChannelNatCompCategory(Simulation& sim, const std::string& modelName,
                         const NDPairList& ndpList);
  void computeE(RNG& rng);
  void count();
};

#endif
