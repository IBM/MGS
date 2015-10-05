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

#ifndef ChannelSchweighoferKDRCompCategory_H
#define ChannelSchweighoferKDRCompCategory_H

#include "Lens.h"
#include "CG_ChannelSchweighoferKDRCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelSchweighoferKDRCompCategory : public CG_ChannelSchweighoferKDRCompCategory, public CountableModel
{
   public:
      ChannelSchweighoferKDRCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();      
};

#endif
