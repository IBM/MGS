// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GranuleMapperTypeDataItem.h"
#include "GranuleMapperType.h"

// Type
const char* GranuleMapperTypeDataItem::_type = "GRANULEMAPPER_TYPE";

// Constructors
GranuleMapperTypeDataItem::GranuleMapperTypeDataItem(GranuleMapperType *data) 
   : _data(data)
{
}


void GranuleMapperTypeDataItem::setInstanceFactory(InstanceFactory* ifp )
{
   GranuleMapperType *ftp = dynamic_cast<GranuleMapperType*>(ifp);
   if(ftp ==0) {
      std::cerr<< "GranuleMapperTypeDataItem:Unable to cast InstanceFactory to GranuleMapperType!"<<std::endl;
      exit(-1);
   }
   setGranuleMapperType(ftp);
}


InstanceFactory* GranuleMapperTypeDataItem::getInstanceFactory() const
{
   return getGranuleMapperType();
}


GranuleMapperTypeDataItem::GranuleMapperTypeDataItem(const GranuleMapperTypeDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void GranuleMapperTypeDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new GranuleMapperTypeDataItem(*this));
}


GranuleMapperTypeDataItem& GranuleMapperTypeDataItem::operator=(const GranuleMapperTypeDataItem& DI)
{
   _data = DI.getGranuleMapperType();
   return(*this);
}


const char* GranuleMapperTypeDataItem::getType() const
{
   return _type;
}


// Singlet methods

GranuleMapperType* GranuleMapperTypeDataItem::getGranuleMapperType() const
{
   return _data;
}


void GranuleMapperTypeDataItem::setGranuleMapperType(GranuleMapperType* i)
{
   _data = i;
}
