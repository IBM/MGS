// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef EpileptorNodeCompCategory_H
#define EpileptorNodeCompCategory_H

#include "Mgs.h"
#include "CG_EpileptorNodeCompCategory.h"

class NDPairList;

class EpileptorNodeCompCategory : public CG_EpileptorNodeCompCategory
{
   public:
      EpileptorNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      std::map<std::pair<int, int>, float> connectionMap;
};

#endif
