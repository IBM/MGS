// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef ChannelSchweighoferKCaCompCategory_H
#define ChannelSchweighoferKCaCompCategory_H

#include "Lens.h"
#include "CG_ChannelSchweighoferKCaCompCategory.h"
#include "../../../../../nti/CountableModel.h"

class NDPairList;

class ChannelSchweighoferKCaCompCategory : public CG_ChannelSchweighoferKCaCompCategory, public CountableModel
{
   public:
      ChannelSchweighoferKCaCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();      
};

#endif
