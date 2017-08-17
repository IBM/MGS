// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef TRIGGERABLEDATAITEM_H
#define TRIGGERABLEDATAITEM_H
#include "Copyright.h"

class Triggerable;
#include <vector>
#include "DataItem.h"

class TriggerableDataItem : public DataItem
{
   public:
      virtual std::vector<Triggerable*> getTriggerables() = 0;
      ~TriggerableDataItem() {}
};
#endif
