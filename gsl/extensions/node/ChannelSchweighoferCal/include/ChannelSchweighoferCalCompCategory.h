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

#ifndef ChannelSchweighoferCalCompCategory_H
#define ChannelSchweighoferCalCompCategory_H

#include "Lens.h"
#include "CG_ChannelSchweighoferCalCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelSchweighoferCalCompCategory : public CG_ChannelSchweighoferCalCompCategory, public CountableModel
{
   public:
      ChannelSchweighoferCalCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void count();   
};

#endif
