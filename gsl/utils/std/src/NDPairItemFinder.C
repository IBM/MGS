// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NDPairItemFinder.h"
#include "NDPairList.h"
#include "NDPair.h"
#include <list>

NDPairList::iterator NDPairItemFinder::find(NDPairList& ndpList, const std::string& item)
{
   NDPairList::iterator it, end = ndpList.end();
   for (it = ndpList.begin(); it != end; it ++) {
      if ((*it)->getName() == item) {
	 return it;
      }
   }
   return it;
}
