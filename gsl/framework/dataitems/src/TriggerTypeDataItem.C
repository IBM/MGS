// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "TriggerTypeDataItem.h"
#include "TriggerType.h"
#include <stdio.h>
#include <stdlib.h>

// Type
const char* TriggerTypeDataItem::_type = "TRIGGER_TYPE";

// Constructors
TriggerTypeDataItem::TriggerTypeDataItem(TriggerType *data) 
   : _data(data)
{
}


void TriggerTypeDataItem::setInstanceFactory(InstanceFactory* ifp )
{
   TriggerType *ftp = dynamic_cast<TriggerType*>(ifp);
   if(ftp ==0) {
      std::cerr<< "TriggerTypeDataItem:Unable to cast InstanceFactory to TriggerType!"<<std::endl;
      exit(-1);
   }
   setTriggerType(ftp);
}


InstanceFactory* TriggerTypeDataItem::getInstanceFactory() const
{
   return getTriggerType();
}


TriggerTypeDataItem::TriggerTypeDataItem(const TriggerTypeDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void TriggerTypeDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new TriggerTypeDataItem(*this));
}


TriggerTypeDataItem& TriggerTypeDataItem::operator=(const TriggerTypeDataItem& DI)
{
   _data = DI.getTriggerType();
   return(*this);
}


const char* TriggerTypeDataItem::getType() const
{
   return _type;
}


// Singlet methods

TriggerType* TriggerTypeDataItem::getTriggerType() const
{
   return _data;
}


void TriggerTypeDataItem::setTriggerType(TriggerType* i)
{
   _data = i;
}
