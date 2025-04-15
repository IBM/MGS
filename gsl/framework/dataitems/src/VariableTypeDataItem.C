// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "VariableTypeDataItem.h"
#include "VariableType.h"
#include <stdio.h>
#include <stdlib.h>

// Type
const char* VariableTypeDataItem::_type = "VARIABLE_TYPE";

// Constructors
VariableTypeDataItem::VariableTypeDataItem(VariableType *data) 
   : _data(data)
{
}


void VariableTypeDataItem::setInstanceFactory(InstanceFactory* ifp )
{
   VariableType *ftp = dynamic_cast<VariableType*>(ifp);
   if(ftp ==0) {
      std::cerr<< "VariableTypeDataItem:Unable to cast InstanceFactory to VariableType!"<<std::endl;
      exit(-1);
   }
   setVariableType(ftp);
}


InstanceFactory* VariableTypeDataItem::getInstanceFactory() const
{
   return getVariableType();
}


VariableTypeDataItem::VariableTypeDataItem(const VariableTypeDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void VariableTypeDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new VariableTypeDataItem(*this));
}


VariableTypeDataItem& VariableTypeDataItem::operator=(const VariableTypeDataItem& DI)
{
   _data = DI.getVariableType();
   return(*this);
}


const char* VariableTypeDataItem::getType() const
{
   return _type;
}


// Singlet methods

VariableType* VariableTypeDataItem::getVariableType() const
{
   return _data;
}


void VariableTypeDataItem::setVariableType(VariableType* i)
{
   _data = i;
}
