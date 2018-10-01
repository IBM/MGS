//=================================================================
//Licensed Materials - Property of IBM
//
//"Restricted Materials of IBM"
//
//BMC-YKT-03-25-2018
//
//(C) Copyright IBM Corp. 2005-2017  All rights reserved
//
//US Government Users Restricted Rights -
//Use, duplication or disclosure restricted by
//GSA ADP Schedule Contract with IBM Corp.
//
//================================================================
#ifndef ChannelCaHVACompCategory_H
#define ChannelCaHVACompCategory_H

#include "Lens.h"
#include "CG_ChannelCaHVACompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelCaHVACompCategory : public CG_ChannelCaHVACompCategory,
                               public CountableModel
{
   public:
      ChannelCaHVACompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeTadj(RNG& rng);
      void count();
};

#endif
