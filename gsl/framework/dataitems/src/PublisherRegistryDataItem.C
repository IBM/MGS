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
void PublisherRegistryDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
