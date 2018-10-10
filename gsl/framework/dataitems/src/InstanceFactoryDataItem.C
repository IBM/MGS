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

#include "InstanceFactoryDataItem.h"
#include "InstanceFactory.h"

// Type
const char* InstanceFactoryDataItem::_type = "INSTANCEFACTORY";

// Constructors
InstanceFactoryDataItem::InstanceFactoryDataItem(InstanceFactory *instanceFactory)
   : _instanceFactory(instanceFactory)
{
}


InstanceFactoryDataItem::InstanceFactoryDataItem(const InstanceFactoryDataItem& DI)
{
   _instanceFactory = DI._instanceFactory;
}


// Utility methods
void InstanceFactoryDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new InstanceFactoryDataItem(*this));
}


InstanceFactoryDataItem& InstanceFactoryDataItem::operator=(const InstanceFactoryDataItem& DI)
{
   _instanceFactory = DI.getInstanceFactory();
   return(*this);
}


const char* InstanceFactoryDataItem::getType() const
{
   return _type;
}


// Singlet methods

InstanceFactory* InstanceFactoryDataItem::getInstanceFactory() const
{
   return _instanceFactory;
}


void InstanceFactoryDataItem::setInstanceFactory(InstanceFactory* s)
{
   _instanceFactory = s;
}


std::string InstanceFactoryDataItem::getString(Error* error) const
{
   return "";
}


InstanceFactoryDataItem::~InstanceFactoryDataItem()
{
}
