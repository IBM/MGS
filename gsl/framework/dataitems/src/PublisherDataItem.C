// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
void PublisherDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
