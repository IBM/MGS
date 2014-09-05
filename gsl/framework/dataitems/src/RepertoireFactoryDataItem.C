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

#include "RepertoireFactoryDataItem.h"
#include "RepertoireFactory.h"

// Type
const char* RepertoireFactoryDataItem::_type = "REPERTOIRE_FACTORY";

// Constructors

RepertoireFactoryDataItem::RepertoireFactoryDataItem()
   : _data(0)
{
}

RepertoireFactoryDataItem::RepertoireFactoryDataItem(std::auto_ptr<RepertoireFactory> data)
{
   _data = data.release();
}


RepertoireFactoryDataItem::RepertoireFactoryDataItem(const RepertoireFactoryDataItem& DI)
   : _data(0)
{
   if (DI._data) {
      std::auto_ptr<RepertoireFactory> dup;
      DI._data->duplicate(dup);
      _data = dup.release();
   }
}

RepertoireFactoryDataItem::~RepertoireFactoryDataItem()
{
   delete _data;
}

// Utility methods
void RepertoireFactoryDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new RepertoireFactoryDataItem(*this)));
}


RepertoireFactoryDataItem& RepertoireFactoryDataItem::operator=(const RepertoireFactoryDataItem& DI)
{

   delete _data;
   if (DI._data) {
      std::auto_ptr<RepertoireFactory> dup;
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


void RepertoireFactoryDataItem::setFactory(std::auto_ptr<RepertoireFactory>& rf, Error* error)
{
   delete _data;
   _data = rf.release();
}
