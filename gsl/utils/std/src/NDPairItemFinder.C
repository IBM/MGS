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
