// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "PublisherDataItem.h"
#include "Publisher.h"

// Type
const char* PublisherDataItem::_type = "PUBLISHER";

// Constructors
PublisherDataItem::PublisherDataItem(Publisher *publisher)
   : _publisher(publisher)
{
}


PublisherDataItem::PublisherDataItem(const PublisherDataItem& DI)
{
   _publisher = DI._publisher;
}


// Utility methods
void PublisherDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new PublisherDataItem(*this));
}


PublisherDataItem& PublisherDataItem::operator=(const PublisherDataItem& DI)
{
   _publisher = DI.getPublisher();
   return(*this);
}


const char* PublisherDataItem::getType() const
{
   return _type;
}


// Singlet methods

Publisher* PublisherDataItem::getPublisher() const
{
   return _publisher;
}


void PublisherDataItem::setPublisher(Publisher* p)
{
   _publisher = p;
}


PublisherDataItem::~PublisherDataItem()
{
}
