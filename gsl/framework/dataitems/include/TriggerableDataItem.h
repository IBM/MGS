// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
