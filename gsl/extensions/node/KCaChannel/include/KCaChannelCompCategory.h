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

#ifndef KCaChannelCompCategory_H
#define KCaChannelCompCategory_H

#include "Lens.h"
#include "CG_KCaChannelCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class KCaChannelCompCategory : public CG_KCaChannelCompCategory, public CountableModel
{
   public:
      KCaChannelCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE_K(RNG& rng);
      void count();      
};

#endif
