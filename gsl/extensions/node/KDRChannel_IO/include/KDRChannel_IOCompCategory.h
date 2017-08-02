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

#ifndef KDRChannel_IOCompCategory_H
#define KDRChannel_IOCompCategory_H

#include "Lens.h"
#include "CG_KDRChannel_IOCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class KDRChannel_IOCompCategory : public CG_KDRChannel_IOCompCategory, public CountableModel
{
   public:
      KDRChannel_IOCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE_KDR(RNG& rng);
      void count();      
};

#endif
