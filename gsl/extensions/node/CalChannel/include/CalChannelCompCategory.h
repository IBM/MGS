// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CalChannelCompCategory_H
#define CalChannelCompCategory_H

#include "Lens.h"
#include "CG_CalChannelCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class CalChannelCompCategory : public CG_CalChannelCompCategory, public CountableModel
{
   public:
      CalChannelCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void count();   
};

#endif
