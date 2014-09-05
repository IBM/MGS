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
void ConstantTypeDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
