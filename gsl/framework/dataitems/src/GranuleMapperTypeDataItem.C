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
void GranuleMapperTypeDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
