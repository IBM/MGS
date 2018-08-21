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
void InstanceFactoryRegistryDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
