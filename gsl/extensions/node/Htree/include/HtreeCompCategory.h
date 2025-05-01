// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef HtreeCompCategory_H
#define HtreeCompCategory_H
/*
@ University of Canterbury 2017-2018. All rights reserved.
*/

#include "Mgs.h"
#include "CG_HtreeCompCategory.h"

class NDPairList;

class HtreeCompCategory : public CG_HtreeCompCategory
{
   public:
      HtreeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
