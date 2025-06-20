// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "TriggerDataItem.h"
#include "Trigger.h"

// Type
const char* TriggerDataItem::_type = "TRIGGER";

// Constructors
TriggerDataItem::TriggerDataItem(Trigger *trigger) 
   : _trigger(trigger)
{
}


TriggerDataItem::TriggerDataItem(const TriggerDataItem& DI)
{
   _trigger = DI._trigger;
}


// Utility methods
void TriggerDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new TriggerDataItem(*this));
}


TriggerDataItem& TriggerDataItem::operator=(const TriggerDataItem& DI)
{
   _trigger = DI.getTrigger();
   return(*this);
}


const char* TriggerDataItem::getType() const
{
   return _type;
}


// Singlet methods

Trigger* TriggerDataItem::getTrigger() const
{
   return _trigger;
}


void TriggerDataItem::setTrigger(Trigger* t)
{
   _trigger = t;
}


TriggerDataItem::~TriggerDataItem()
{
}
