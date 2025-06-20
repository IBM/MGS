// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "StructTypeDataItem.h"
#include "StructType.h"
#include <stdio.h>
#include <stdlib.h>

// Type
const char* StructTypeDataItem::_type = "STRUCT_TYPE";

// Constructors
StructTypeDataItem::StructTypeDataItem(StructType *data) 
   : _data(data)
{
}


void StructTypeDataItem::setInstanceFactory(InstanceFactory* ifp )
{
   StructType *ftp = dynamic_cast<StructType*>(ifp);
   if(ftp ==0) {
      std::cerr<< "StructTypeDataItem:Unable to cast InstanceFactory to StructType!"<<std::endl;
      exit(-1);
   }
   setStructType(ftp);
}


InstanceFactory* StructTypeDataItem::getInstanceFactory() const
{
   return getStructType();
}


StructTypeDataItem::StructTypeDataItem(const StructTypeDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void StructTypeDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new StructTypeDataItem(*this));
}


StructTypeDataItem& StructTypeDataItem::operator=(const StructTypeDataItem& DI)
{
   _data = DI.getStructType();
   return(*this);
}


const char* StructTypeDataItem::getType() const
{
   return _type;
}


// Singlet methods

StructType* StructTypeDataItem::getStructType() const
{
   return _data;
}


void StructTypeDataItem::setStructType(StructType* i)
{
   _data = i;
}
