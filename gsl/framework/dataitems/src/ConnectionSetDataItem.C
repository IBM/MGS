// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ConnectionSetDataItem.h"

const char* ConnectionSetDataItem::_type = "CONNECTION_SET";

ConnectionSetDataItem& ConnectionSetDataItem::operator=(const ConnectionSetDataItem& DI)
{
   _data = DI.getConnectionSet();
   return(*this);
}


ConnectionSetDataItem::ConnectionSetDataItem(ConnectionSet *data)
   : _data(data)
{
}


ConnectionSetDataItem::ConnectionSetDataItem(const ConnectionSetDataItem& DI)
{
   _data = DI._data;
}


void ConnectionSetDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new ConnectionSetDataItem(*this)));
}


const char* ConnectionSetDataItem::getType() const
{
   return _type;
}


// Singlet methods

ConnectionSet* ConnectionSetDataItem::getConnectionSet(Error* error) const
{
   return _data;
}


void ConnectionSetDataItem::setConnectionSet(ConnectionSet* i, Error* error)
{
   _data = i;
}
