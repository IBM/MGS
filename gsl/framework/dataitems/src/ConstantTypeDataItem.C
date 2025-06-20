// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ConstantTypeDataItem.h"
#include "ConstantType.h"
#include <stdio.h>
#include <stdlib.h>

// Type
const char* ConstantTypeDataItem::_type = "CONSTANT_TYPE";

// Constructors
ConstantTypeDataItem::ConstantTypeDataItem(ConstantType *data) 
   : _data(data)
{
}


void ConstantTypeDataItem::setInstanceFactory(InstanceFactory* ifp )
{
   ConstantType *ftp = dynamic_cast<ConstantType*>(ifp);
   if(ftp ==0) {
      std::cerr<< "ConstantTypeDataItem:Unable to cast InstanceFactory to ConstantType!"<<std::endl;
      exit(-1);
   }
   setConstantType(ftp);
}


InstanceFactory* ConstantTypeDataItem::getInstanceFactory() const
{
   return getConstantType();
}


ConstantTypeDataItem::ConstantTypeDataItem(const ConstantTypeDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void ConstantTypeDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new ConstantTypeDataItem(*this));
}


ConstantTypeDataItem& ConstantTypeDataItem::operator=(const ConstantTypeDataItem& DI)
{
   _data = DI.getConstantType();
   return(*this);
}


const char* ConstantTypeDataItem::getType() const
{
   return _type;
}


// Singlet methods

ConstantType* ConstantTypeDataItem::getConstantType() const
{
   return _data;
}


void ConstantTypeDataItem::setConstantType(ConstantType* i)
{
   _data = i;
}
