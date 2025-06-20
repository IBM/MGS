// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NDPAIRITEMFINDER_H
#define NDPAIRITEMFINDER_H
#include "Copyright.h"

#include "NDPairList.h"
#include <string>
#include <list>

class NDPairItemFinder
{
   public:
      NDPairItemFinder() {};

      NDPairList::iterator find(NDPairList& ndpList, const std::string& item);
      ~NDPairItemFinder() {};
};
#endif
