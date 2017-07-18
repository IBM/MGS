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

#include "EdgeTypeDataItem.h"
#include "EdgeType.h"

// Type
const char* EdgeTypeDataItem::_type = "EDGE_TYPE";

// Constructors
EdgeTypeDataItem::EdgeTypeDataItem(EdgeType *data)
   : _data(data)
{
}


EdgeTypeDataItem::EdgeTypeDataItem(const EdgeTypeDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void EdgeTypeDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new EdgeTypeDataItem(*this));
}


EdgeTypeDataItem& EdgeTypeDataItem::operator=(const EdgeTypeDataItem& DI)
{
   _data = DI.getEdgeType();
   return(*this);
}


const char* EdgeTypeDataItem::getType() const
{
   return _type;
}


// Singlet methods

EdgeType* EdgeTypeDataItem::getEdgeType() const
{
   return _data;
}


void EdgeTypeDataItem::setEdgeType(EdgeType* i)
{
   _data = i;
}
