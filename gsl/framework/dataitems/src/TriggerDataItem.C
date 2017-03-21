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
void TriggerDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
