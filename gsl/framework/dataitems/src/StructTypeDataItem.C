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
void StructTypeDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
