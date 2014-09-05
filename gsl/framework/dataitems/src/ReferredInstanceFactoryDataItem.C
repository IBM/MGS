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

#include "ReferredInstanceFactoryDataItem.h"
#include "InstanceFactory.h"

// Type
const char* ReferredInstanceFactoryDataItem::_type = "REFERREDINSTANCEFACTORY";

// Constructors
ReferredInstanceFactoryDataItem::ReferredInstanceFactoryDataItem(InstanceFactory *instanceFactory)
   : _instanceFactory(instanceFactory)
{
}


ReferredInstanceFactoryDataItem::ReferredInstanceFactoryDataItem(const ReferredInstanceFactoryDataItem& DI)
{
   _instanceFactory = DI._instanceFactory;
}


// Utility methods
void ReferredInstanceFactoryDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new ReferredInstanceFactoryDataItem(*this));
}


ReferredInstanceFactoryDataItem& ReferredInstanceFactoryDataItem::operator=(const ReferredInstanceFactoryDataItem& DI)
{
   return assign(DI);
}


ReferredInstanceFactoryDataItem& ReferredInstanceFactoryDataItem::assign(const ReferredInstanceFactoryDataItem& DI)
{
   _instanceFactory = DI.getInstanceFactory();
   return(*this);
}


const char* ReferredInstanceFactoryDataItem::getType() const
{
   return _type;
}


// Singlet methods

InstanceFactory* ReferredInstanceFactoryDataItem::getInstanceFactory() const
{
   return _instanceFactory;
}


void ReferredInstanceFactoryDataItem::setInstanceFactory(InstanceFactory* s)
{
   _instanceFactory = s;
}


ReferredInstanceFactoryDataItem::~ReferredInstanceFactoryDataItem()
{
}
