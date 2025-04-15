// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "RepertoireFactoryDataItem.h"
#include "RepertoireFactory.h"

// Type
const char* RepertoireFactoryDataItem::_type = "REPERTOIRE_FACTORY";

// Constructors

RepertoireFactoryDataItem::RepertoireFactoryDataItem()
   : _data(0)
{
}

RepertoireFactoryDataItem::RepertoireFactoryDataItem(std::unique_ptr<RepertoireFactory> data)
{
   _data = data.release();
}


RepertoireFactoryDataItem::RepertoireFactoryDataItem(const RepertoireFactoryDataItem& DI)
   : _data(0)
{
   if (DI._data) {
      std::unique_ptr<RepertoireFactory> dup;
      DI._data->duplicate(dup);
      _data = dup.release();
   }
}

RepertoireFactoryDataItem::~RepertoireFactoryDataItem()
{
   delete _data;
}

// Utility methods
void RepertoireFactoryDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new RepertoireFactoryDataItem(*this)));
}


RepertoireFactoryDataItem& RepertoireFactoryDataItem::operator=(const RepertoireFactoryDataItem& DI)
{

   delete _data;
   if (DI._data) {
      std::unique_ptr<RepertoireFactory> dup;
      DI._data->duplicate(dup);
      _data = dup.release();
   }
   return(*this);
}


const char* RepertoireFactoryDataItem::getType() const
{
   return _type;
}


RepertoireFactory* RepertoireFactoryDataItem::getFactory(Error* error) const
{
   return _data;
}


void RepertoireFactoryDataItem::setFactory(std::unique_ptr<RepertoireFactory>& rf, Error* error)
{
   delete _data;
   _data = rf.release();
}
