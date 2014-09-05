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


void ConnectionSetDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
