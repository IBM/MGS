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

#ifndef NaChannelCompCategory_H
#define NaChannelCompCategory_H

#include "Lens.h"
#include "CG_NaChannelCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class NaChannelCompCategory : public CG_NaChannelCompCategory, public CountableModel
{
   public:
      NaChannelCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE_Na(RNG& rng);
      void count();      
};

#endif
