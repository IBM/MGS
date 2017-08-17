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

#ifndef KDRChannel_AISCompCategory_H
#define KDRChannel_AISCompCategory_H

#include "Lens.h"
#include "CG_KDRChannel_AISCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class KDRChannel_AISCompCategory : public CG_KDRChannel_AISCompCategory, public CountableModel
{
   public:
      KDRChannel_AISCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE_KDR(RNG& rng);
      void count();      
};

#endif
