// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef ChannelHayHCNCompCategory_H
#define ChannelHayHCNCompCategory_H

#include "Lens.h"
#include "CG_ChannelHayHCNCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelHayHCNCompCategory : public CG_ChannelHayHCNCompCategory, public CountableModel
{
   public:
      ChannelHayHCNCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void count();      
};

#endif
