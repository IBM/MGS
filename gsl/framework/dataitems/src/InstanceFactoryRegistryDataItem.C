// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "InstanceFactoryRegistryDataItem.h"
#include "InstanceFactoryRegistry.h"

// Type
const char* InstanceFactoryRegistryDataItem::_type = "INSTANCEFACTORYREGISTRY";

// Constructors
InstanceFactoryRegistryDataItem::InstanceFactoryRegistryDataItem(InstanceFactoryRegistry *ifReg) 
   : _ifReg(ifReg)
{
}


InstanceFactoryRegistryDataItem::InstanceFactoryRegistryDataItem(const InstanceFactoryRegistryDataItem& DI)
{
   _ifReg = DI._ifReg;
}


// Utility methods
void InstanceFactoryRegistryDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new InstanceFactoryRegistryDataItem(*this));
}


InstanceFactoryRegistryDataItem& InstanceFactoryRegistryDataItem::operator=(const InstanceFactoryRegistryDataItem& DI)
{
   _ifReg = DI.getInstanceFactoryRegistry();
   return(*this);
}


const char* InstanceFactoryRegistryDataItem::getType() const
{
   return _type;
}


// Singlet methods

InstanceFactoryRegistry* InstanceFactoryRegistryDataItem::getInstanceFactoryRegistry() const
{
   return _ifReg;
}


void InstanceFactoryRegistryDataItem::setInstanceFactoryRegistry(InstanceFactoryRegistry* ifr)
{
   _ifReg = ifr;
}


InstanceFactoryRegistryDataItem::~InstanceFactoryRegistryDataItem()
{
}


std::string InstanceFactoryRegistryDataItem::getString(Error* error) const
{
   return "";
}
