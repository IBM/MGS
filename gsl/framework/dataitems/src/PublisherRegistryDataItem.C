// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "PublisherRegistryDataItem.h"
#include "PublisherRegistry.h"

// Type
const char* PublisherRegistryDataItem::_type = "PUBLISHERREGISTRY";

// Constructors
PublisherRegistryDataItem::PublisherRegistryDataItem(PublisherRegistry *pubReg) 
   : _pubReg(pubReg)
{
}


PublisherRegistryDataItem::PublisherRegistryDataItem(const PublisherRegistryDataItem& DI)
{
   _pubReg = DI._pubReg;
}


// Utility methods
void PublisherRegistryDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new PublisherRegistryDataItem(*this));
}


PublisherRegistryDataItem& PublisherRegistryDataItem::operator=(const PublisherRegistryDataItem& DI)
{
   _pubReg = DI.getPublisherRegistry();
   return(*this);
}


const char* PublisherRegistryDataItem::getType() const
{
   return _type;
}


// Singlet methods

PublisherRegistry* PublisherRegistryDataItem::getPublisherRegistry() const
{
   return _pubReg;
}


void PublisherRegistryDataItem::setPublisherRegistry(PublisherRegistry* pr)
{
   _pubReg = pr;
}


PublisherRegistryDataItem::~PublisherRegistryDataItem()
{
}


std::string PublisherRegistryDataItem::getString(Error* error) const
{
   return "";
}
