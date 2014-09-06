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

#ifndef ChannelHayCaHVACompCategory_H
#define ChannelHayCaHVACompCategory_H

#include "Lens.h"
#include "CG_ChannelHayCaHVACompCategory.h"
#include "../../../../../nti/CountableModel.h"

class NDPairList;

class ChannelHayCaHVACompCategory : public CG_ChannelHayCaHVACompCategory, public CountableModel
{
   public:
      ChannelHayCaHVACompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void count();   
};

#endif
