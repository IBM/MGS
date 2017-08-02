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
