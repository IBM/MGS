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
