// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
void ReferredInstanceFactoryDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
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
