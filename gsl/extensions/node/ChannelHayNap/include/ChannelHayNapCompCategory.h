// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef ChannelHayNapCompCategory_H
#define ChannelHayNapCompCategory_H

#include "Lens.h"
#include "CG_ChannelHayNapCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelHayNapCompCategory : public CG_ChannelHayNapCompCategory, public CountableModel
{
   public:
      ChannelHayNapCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
