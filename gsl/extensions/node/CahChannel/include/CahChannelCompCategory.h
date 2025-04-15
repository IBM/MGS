// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CahChannelCompCategory_H
#define CahChannelCompCategory_H

#include "Lens.h"
#include "CG_CahChannelCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class CahChannelCompCategory : public CG_CahChannelCompCategory, public CountableModel
{
   public:
      CahChannelCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void count();   
};

#endif
